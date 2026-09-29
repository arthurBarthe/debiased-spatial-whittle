"""
Neural-network-based expected periodogram estimation via the debiased Whittle
likelihood.

This module provides frequency samplers, a dataset of zero-padded tapered
periodograms, and a trainer that optimises a neural network to directly
predict ln(EP) from model parameters and frequency coordinates.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()


def _mirror_neg_freqs(freq_vals):
    """Enforce the symmetry EP(f) = EP(-f) by folding negative first-axis
    frequencies onto their positive counterpart.

    For any row where the first coordinate ``f1`` is negative, all coordinates
    are negated so the NN is only ever evaluated at ``f1 >= 0``.

    Parameters
    ----------
    freq_vals : torch.Tensor
        Frequency coordinates, shape (B, d).

    Returns
    -------
    torch.Tensor
        Mirrored frequency coordinates, same shape as ``freq_vals``.
    """
    f1 = freq_vals[:, 0]
    neg = f1 < 0
    mirrored = freq_vals.clone()
    mirrored[neg] = -mirrored[neg]
    # Wrap the Nyquist frequency: -0.5 negates to 0.5, which is equivalent to
    # -0.5 in an even-size DFT and absent from the fftfreq grid.
    mirrored = torch.where(mirrored >= 0.5, mirrored - 1.0, mirrored)
    return mirrored


class FrequencySampler:
    """Base class for sampling Fourier frequency indices.

    Parameters
    ----------
    n_freq_total : int
        Total number of frequencies (flat index space).
    n_freq_per_sample : int
        Number of frequencies to draw per call.
    padded_n : tuple
        Shape of the padded frequency grid, e.g. (2*n1, ..., 2*nd).
    """

    def __init__(self, n_freq_total, n_freq_per_sample, padded_n):
        self.n_freq_total = n_freq_total
        self.n_freq_per_sample = n_freq_per_sample
        self.padded_n = padded_n

    def sample(self):
        """Return a 1-D tensor of flat frequency indices."""
        raise NotImplementedError


class UniformFrequencySampler(FrequencySampler):
    """Uniformly random frequency sampling without replacement."""

    def sample(self):
        idx = torch.randperm(self.n_freq_total)[:self.n_freq_per_sample]
        return idx


class InverseRadiusFrequencySampler(FrequencySampler):
    """Sample frequencies with probability proportional to 1/(||f|| + eps).

    Low frequencies (near the origin) are sampled more frequently.

    Parameters
    ----------
    n_freq_total : int
        Total number of frequencies.
    n_freq_per_sample : int
        Number of frequencies to draw per call.
    padded_n : tuple
        Shape of the padded frequency grid.
    eps : float, optional
        Regularisation constant to avoid division by zero.
        Default is 0.01.
    """

    def __init__(self, n_freq_total, n_freq_per_sample, padded_n, eps=0.01):
        super().__init__(n_freq_total, n_freq_per_sample, padded_n)
        self.eps = eps

        ndim = len(padded_n)
        freq_1d = [torch.fft.fftfreq(s, 1.0) for s in padded_n]
        mesh = torch.meshgrid(*freq_1d, indexing="ij")
        freq_grid = torch.stack(mesh, axis=-1)  # (*padded_n, d)
        norms = torch.norm(freq_grid, dim=-1)  # (*padded_n,)
        weights = 1.0 / (norms.flatten() + eps)
        self.prob = weights / weights.sum()

    def sample(self):
        idx = torch.multinomial(
            self.prob, self.n_freq_per_sample, replacement=False
        )
        return idx


def _flat_to_multi(idx, padded_n):
    """Convert flat indices to multi-indices for an N-dimensional grid.

    Parameters
    ----------
    idx : torch.Tensor
        1-D tensor of flat indices.
    padded_n : tuple
        Shape of the grid.

    Returns
    -------
    torch.Tensor
        Tensor of shape (len(idx), len(padded_n)).
    """
    multi = []
    remaining = idx
    # Dimensions go from slowest to fastest, so we peel from the end
    for n in reversed(padded_n):
        multi.append(remaining % n)
        remaining = remaining // n
    return torch.stack(list(reversed(multi)), dim=1)


class PaddedPeriodogramDataset(Dataset):
    """Dataset of zero-padded tapered periodograms with per-sample frequency subsampling.

    Computes the periodogram of each sample (via the provided ``periodogram``
    object, which should have ``fold=False`` so the output matches the unfolded
    expected periodogram on the ``(2*n1, ..., 2*nd)`` grid) and draws a random
    set of Fourier frequencies independently for each sample using a
    :class:`FrequencySampler`.  Call ``resample_frequencies`` at the end of each
    epoch to draw a fresh set.

    Parameters
    ----------
    grid : RectangularGrid
        The sampling grid.
    samples : list[SampleOnRectangularGrid]
        K samples on the grid.
    param_values : list[tuple]
        K tuples of parameter values.
    periodogram : Periodogram
        Periodogram object used to compute each sample's periodogram. Its
        ``fold`` attribute controls the output shape.
    freq_fraction : float, optional
        Fraction of Fourier frequencies to use per sample. Default 1.0.
    freq_sampler : FrequencySampler, optional
        Custom frequency sampler. If None, :class:`UniformFrequencySampler`
        is used.
    """

    def __init__(self, grid, samples, param_values, periodogram,
                 freq_fraction=1.0, freq_sampler=None):
        self.grid = grid
        self.grid_n = grid.n
        self.padded_n = tuple(2 * s for s in grid.n)
        self.n_freq_total = int(np.prod(self.padded_n))
        self.ndim = grid.ndim
        self.freq_fraction = freq_fraction
        self.n_freq_per_sample = max(1, int(self.n_freq_total * freq_fraction))
        self.n_samples = len(samples)

        if freq_sampler is None:
            freq_sampler = UniformFrequencySampler(
                self.n_freq_total, self.n_freq_per_sample, self.padded_n
            )
        self.freq_sampler = freq_sampler

        self.periodograms = [periodogram(s) for s in samples]

        self.param_values = torch.as_tensor(param_values, dtype=torch.float64)

        self._freq_multi_indices = [None] * self.n_samples
        self.resample_frequencies()

    def resample_frequencies(self):
        """Draw a fresh random set of frequencies for each sample."""
        for k in range(self.n_samples):
            if self.freq_fraction < 1.0:
                idx = self.freq_sampler.sample()
            else:
                idx = torch.arange(self.n_freq_total)
            self._freq_multi_indices[k] = _flat_to_multi(idx, self.padded_n)

    def __len__(self):
        return self.n_freq_per_sample * self.n_samples

    def __getitem__(self, idx):
        k = idx // self.n_freq_per_sample
        f = idx % self.n_freq_per_sample
        freq_index = self._freq_multi_indices[k][f]
        params = self.param_values[k]
        p_val = self.periodograms[k][tuple(freq_index.tolist())]
        return tuple(freq_index.tolist()), params, p_val


class NnEpTrainer:
    """Trainer for a neural network that directly predicts ln(EP).

    The NN takes ``(a + d)`` inputs — ``a`` model parameter values followed
    by ``d`` frequency coordinates — and outputs a scalar ``ln(EP)``.
    Training uses the debiased Whittle loss
    ``mean(ln(EP) + P / EP)`` where ``P`` is the tapered periodogram.

    Parameters
    ----------
    nn_module : torch.nn.Module
        Neural network mapping ``(a + d)``-dimensional input to scalar
        ``ln(EP)``.
    dataset : PaddedPeriodogramDataset
        Dataset providing (freq_index, param_values, periodogram_value).
    freq_grid : torch.Tensor
        Frequency coordinate grid of shape ``(*padded_n, d)``, in FFT order.
    device : str, optional
        Device for training. Default is "cuda:0" if available, else "cpu".
    learning_rate : float, optional
        Initial learning rate. Default 1e-2.
    batch_size : int, optional
        Batch size. Default 8192.
    n_epochs : int, optional
        Number of epochs. Default 60.
    lr_schedule : list, optional
        List of (epoch, lr) tuples for learning rate decay.
        Default decays at epochs 20 and 40.
    """

    def __init__(
        self,
        nn_module,
        dataset,
        freq_grid,
        device=None,
        learning_rate=1e-2,
        batch_size=8192,
        n_epochs=60,
        lr_schedule=None,
    ):
        self.nn_module = nn_module
        self.dataset = dataset
        self.freq_grid = freq_grid
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr_schedule = lr_schedule or [(20, 1e-3), (40, 1e-4)]

        self.optimizer = torch.optim.Adam(
            nn_module.parameters(), lr=learning_rate
        )

        self.padded_n = dataset.padded_n
        self.d = dataset.ndim
        self.N_freq = dataset.n_freq_total

    def fit(self, verbose=True):
        """Train the NN weights.

        Returns
        -------
        dict
            Training history with 'loss' key.
        """
        gen = torch.Generator(device=self.device)
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True, generator=gen,
        )

        history = {"loss": []}

        for epoch in range(self.n_epochs):
            for trigger_epoch, lr in self.lr_schedule:
                if epoch == trigger_epoch:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

            epoch_loss = 0.0
            n_batches = 0
            self.nn_module.train()
            for batch in dataloader:
                freq_indices, params, p_values = batch

                # (B, d) multi-index tensor
                batch_indices = torch.stack(
                    [torch.as_tensor(fi, dtype=torch.long) for fi in freq_indices],
                    dim=1,
                ).to(self.device)

                # Look up frequency coordinates: (B, d)
                # Use advanced indexing on freq_grid (*padded_n, d)
                idx_tuple = tuple(batch_indices[:, dim] for dim in range(self.d))
                freq_vals = self.freq_grid[idx_tuple]  # (B, d)
                freq_vals = _mirror_neg_freqs(freq_vals)

                params_dev = params.to(dtype=torch.float64, device=self.device)
                nn_input = torch.cat([params_dev, freq_vals], dim=1)

                log_ep = self.nn_module(nn_input).squeeze(-1)
                ep_pred = torch.exp(log_ep)

                p_dev = p_values.to(dtype=torch.float64, device=self.device)

                loss = torch.mean(log_ep + p_dev / ep_pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.dataset.resample_frequencies()

            epoch_loss /= n_batches
            history["loss"].append(epoch_loss)
            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

        return history

    def predict_ep(self, params, chunk_size=8192 * 4):
        """Evaluate the NN on the full frequency grid.

        Parameters
        ----------
        params : torch.Tensor
            1-D tensor of parameter values, shape (a,).
        chunk_size : int, optional
            Chunk size to avoid OOM. Default 8192.

        Returns
        -------
        torch.Tensor
            EP values in FFT order, shape ``padded_n``.
        """
        freq_flat = self.freq_grid.reshape(self.N_freq, self.d)
        a = len(params)
        params = params.to(dtype=torch.float64, device=self.device)

        pred_parts = []
        with torch.no_grad():
            self.nn_module.eval()
            for start in range(0, self.N_freq, chunk_size):
                end = min(start + chunk_size, self.N_freq)
                freq_chunk = _mirror_neg_freqs(freq_flat[start:end].to(self.device))
                param_rows = params.unsqueeze(0).expand(end - start, a)
                nn_input = torch.cat(
                    [param_rows, freq_chunk], dim=1
                )
                pred_parts.append(self.nn_module(nn_input).squeeze(-1).cpu())
        log_ep = torch.cat(pred_parts).reshape(*self.padded_n)
        return torch.exp(log_ep)
