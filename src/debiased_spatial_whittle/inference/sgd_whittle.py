"""
Stochastic Gradient Descent implementation for Debiased Whittle Likelihood.

This module provides:
1. FrequencyIndexDataset - A PyTorch Dataset class for frequency indices
2. StochasticDebiasedWhittle - Stochastic version of Debiased Whittle Likelihood
3. SGDTrainer - Training algorithm using torch.optim.SGD

Note: This implementation uses the existing FFT-based periodogram and expected periodogram
implementations, but extracts values at specific frequency indices for stochastic optimization.
"""

import torch
import numpy as np
from itertools import product
from typing import Tuple, List, Optional, Union
from torch.utils.data import Dataset, DataLoader

from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.models.base import CovarianceModel, ModelInterface, ReparameterizedModel
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram

xp = BackendManager.get_backend()
fftshift, ifftshift = BackendManager.get_fftshift_methods()


class FrequencyIndexDataset(Dataset):
    """
    A PyTorch Dataset class that returns frequency indices for a grid.
    
    For a grid of shape (n1, n2, ..., nd), this dataset provides all possible
    frequency indices (i1, i2, ..., id) where 0 <= ij < nj for each dimension j.
    
    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Shape of the grid (e.g., (256, 512) for a 2D grid)
    
    Examples
    --------
    >>> dataset = FrequencyIndexDataset((32, 64))
    >>> len(dataset)
    2048
    >>> dataset[0]
    (0, 0)
    >>> dataset[63]
    (0, 63)
    >>> dataset[64]
    (1, 0)
    """
    
    def __init__(self, grid_shape: tuple[int, ...]):
        self.grid_shape = grid_shape
        self.n_points = np.prod(grid_shape)
        
        # Precompute all indices for efficiency
        self._indices = []
        if len(grid_shape) == 1:
            for i in range(grid_shape[0]):
                self._indices.append((i,))
        elif len(grid_shape) == 2:
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    self._indices.append((i, j))
        elif len(grid_shape) == 3:
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    for k in range(grid_shape[2]):
                        self._indices.append((i, j, k))
        else:
            # Generic implementation for higher dimensions
            from itertools import product
            ranges = [range(s) for s in grid_shape]
            for idx in product(*ranges):
                self._indices.append(idx)
    
    def __len__(self) -> int:
        return self.n_points
    
    def __getitem__(self, idx: int) -> tuple[int, ...]:
        """
        Returns the frequency index at position idx.
        
        Parameters
        ----------
        idx : int
            Index in the dataset
            
        Returns
        -------
        tuple[int, ...]
            Frequency index tuple (i1, i2, ..., id)
        """
        return self._indices[idx]
    
    def get_batch(self, indices: List[int]) -> List[tuple[int, ...]]:
        """
        Get a batch of frequency indices.
        
        Parameters
        ----------
        indices : List[int]
            List of dataset indices
            
        Returns
        -------
        List[tuple[int, ...]]
            List of frequency index tuples
        """
        return [self[i] for i in indices]


class StochasticDebiasedWhittle:
    """
    Stochastic version of the Debiased Whittle Likelihood.
    
    This computes the likelihood for a batch of frequency indices by using the existing
    FFT-based periodogram and expected periodogram implementations, but only extracting
    the values at the specified indices for the batch.
    
    Parameters
    ----------
    grid : RectangularGrid
        The grid on which the data is sampled
    periodogram : Periodogram
        Periodogram object for computing periodogram values
    expected_periodogram : ExpectedPeriodogram
        Expected periodogram object

    Attributes
    ----------
    grid : RectangularGrid
        The sampling grid
    """

    def __init__(
        self,
        grid: RectangularGrid,
        periodogram: Periodogram,
        expected_periodogram: ExpectedPeriodogram
    ):
        self.grid = grid
        self.periodogram = periodogram
        self.expected_periodogram = expected_periodogram
        
    def compute_periodogram_at_indices(
        self,
        sample: SampleOnRectangularGrid,
        indices: List[tuple[int, ...]]
    ) -> torch.Tensor:
        """
        Compute the periodogram at specific frequency indices.
        
        This uses the existing FFT-based periodogram implementation and extracts
        values at the specified indices.
        
        Parameters
        ----------
        sample : SampleOnRectangularGrid
            The data sample
        indices : List[tuple[int, ...]]
            List of frequency indices
            
        Returns
        -------
        torch.Tensor
            Periodogram values at the specified indices, shape (batch_size,)
        """
        # Get the full periodogram using existing FFT-based implementation
        full_periodogram = self.periodogram(sample)
        
        # Convert to torch tensor if needed
        if not isinstance(full_periodogram, torch.Tensor):
            full_periodogram = torch.tensor(full_periodogram, dtype=torch.float64)
        
        # Extract values at the specified indices using vectorized advanced indexing
        idx_array = torch.as_tensor(indices, dtype=torch.long)
        idx_per_dim = torch.unbind(idx_array, dim=1)
        periodogram_values = full_periodogram[idx_per_dim].real

        return periodogram_values
    
    def compute_expected_periodogram_at_indices(
        self,
        model: CovarianceModel,
        indices: List[tuple[int, ...]]
    ) -> torch.Tensor:
        """
        Compute the expected periodogram at specific frequency indices.
        
        This uses the existing FFT-based expected periodogram implementation and extracts
        values at the specified indices. Note that this will not support autograd through
        the model parameters since the FFT-based implementation doesn't support it.
        
        Parameters
        ----------
        model : CovarianceModel
            The covariance model
        indices : List[tuple[int, ...]]
            List of frequency indices
            
        Returns
        -------
        torch.Tensor
            Expected periodogram values at the specified indices, shape (batch_size,)
        """
        # Get the full expected periodogram using existing FFT-based implementation
        full_ep = self.expected_periodogram(model)
        
        # Convert to torch tensor if needed
        if not isinstance(full_ep, torch.Tensor):
            full_ep = torch.tensor(full_ep, dtype=torch.float64)
        
        # Extract values at the specified indices using vectorized advanced indexing
        idx_array = torch.as_tensor(indices, dtype=torch.long)
        idx_per_dim = torch.unbind(idx_array, dim=1)
        ep_values = full_ep[idx_per_dim].real

        return ep_values
    
    def compute_expected_periodogram_at_indices_autograd(
        self,
        model: CovarianceModel,
        indices: List[tuple[int, ...]]
    ) -> torch.Tensor:
        """
        Compute the expected periodogram at specific frequency indices with autograd.

        This method builds the expected periodogram through the existing FFT-based
        ``compute_ep`` pipeline, while keeping an autograd graph connecting the free
        model parameters to the output. The autocovariance is evaluated on the full
        grid of lags in a single vectorized call to ``model.compute`` (structured like
        the ``func`` in ``ModelInterface.jacobian``), then folded and transformed by
        ``ExpectedPeriodogram.compute_ep``. The values at the requested frequency
        indices are finally extracted via vectorized advanced indexing.

        This replaces a slow per-frequency, per-lag direct-evaluation loop while
        preserving gradients: free params (leaf tensors, ``requires_grad=True``)
        -> ``model.compute`` -> ``ifftshift`` -> ``compute_ep`` (pad + fftn + real)
        -> extracted values.

        Parameters
        ----------
        model : CovarianceModel
            The covariance model
        indices : List[tuple[int, ...]]
            List of frequency indices

        Returns
        -------
        torch.Tensor
            Expected periodogram values at the specified indices, shape (batch_size,)
        """
        # Lags for the autocovariance grid: shape (ndim, 2*n1-1, ..., 2*nd-1)
        lags = self.grid.lags_unique
        if not isinstance(lags, torch.Tensor):
            lags = torch.as_tensor(lags, dtype=torch.float64)
        ndim = self.grid.ndim

        # Build acv with an autograd graph through the free parameters,
        # mirroring the func pattern in ModelInterface.jacobian (base.py:235).
        param_names = model.free_parameter_names
        all_param_names = model.parameter_names
        param_values = model.get_parameters(param_names)  # leaf tensors, requires_grad=True

        # Trailing dim expected by compute (as ModelInterface.__call__ adds)
        lags_expanded = lags.unsqueeze(-1)

        def func(*args):
            full_args = []
            for pname in all_param_names:
                if pname in param_names:
                    full_args.append(args[param_names.index(pname)])
                else:
                    full_args.append(model.parameters[all_param_names.index(pname)])
            return model.compute(lags_expanded, *full_args)

        acv = func(*param_values)
        # Squeeze the trailing dim added for compute (as ModelInterface.__call__ does)
        if acv.shape[-1] == 1:
            acv = acv.squeeze(-1)
        # ifftshift the spatial dims (as grid.autocov does)
        acv = ifftshift(acv, list(range(ndim)))

        # Expected periodogram via existing FFT-based compute_ep
        full_ep = self.expected_periodogram.compute_ep(
            acv, self.expected_periodogram.periodogram.fold
        )
        if not isinstance(full_ep, torch.Tensor):
            full_ep = torch.as_tensor(full_ep, dtype=torch.float64)

        # Vectorized extraction at the batch indices
        idx_array = torch.as_tensor(indices, dtype=torch.long)
        idx_per_dim = torch.unbind(idx_array, dim=1)
        ep_values = full_ep[idx_per_dim].real
        return ep_values

    def compute_expected_periodogram_at_indices_direct(
        self,
        model: CovarianceModel,
        indices: List[tuple[int, ...]]
    ) -> torch.Tensor:
        """
        Compute the expected periodogram at specific frequency indices directly,
        without FFT.

        This mirrors the folding stage of ``ExpectedPeriodogram.compute_ep`` (spatial
        kernel weighting + fold over the ``2**ndim`` blocks) but replaces the FFT with
        a vectorized evaluation of the DFT at the requested frequency indices only:

            EP(k) = Re( sum_m folded[m] * exp(-2 pi i * k^T m / n) )

        where ``m`` ranges over the grid multi-indices and ``k`` over the requested
        frequency indices. The sum is computed as a batched matrix product between a
        ``(B, M)`` phase matrix and the ``(M,)`` folded array, so there are no Python
        loops over frequencies or lags. Gradients flow through the model free
        parameters as in the autograd (FFT-based) method.

        Parameters
        ----------
        model : CovarianceModel
            The covariance model
        indices : List[tuple[int, ...]]
            List of frequency indices

        Returns
        -------
        torch.Tensor
            Expected periodogram values at the specified indices, shape (batch_size,)
        """
        ndim = self.grid.ndim
        n = self.grid.n

        # --- Autocovariance with an autograd graph through the free parameters ---
        lags = self.grid.lags_unique
        if not isinstance(lags, torch.Tensor):
            lags = torch.as_tensor(lags, dtype=torch.float64)

        param_names = model.free_parameter_names
        all_param_names = model.parameter_names
        param_values = model.get_parameters(param_names)

        lags_expanded = lags.unsqueeze(-1)

        def func(*args):
            full_args = []
            for pname in all_param_names:
                if pname in param_names:
                    full_args.append(args[param_names.index(pname)])
                else:
                    full_args.append(model.parameters[all_param_names.index(pname)])
            return model.compute(lags_expanded, *full_args)

        acv = func(*param_values)
        if acv.shape[-1] == 1:
            acv = acv.squeeze(-1)
        acv = ifftshift(acv, list(range(ndim)))

        # --- Spatial kernel weighting (same as compute_ep, d=(0,0)) ---
        cg = self.grid.spatial_kernel(self.expected_periodogram.taper)
        if not isinstance(cg, torch.Tensor):
            cg = torch.as_tensor(cg, dtype=acv.dtype)
        cbar = cg * acv

        # --- Fold over the 2**ndim blocks (structural, independent of batch/grid) ---
        if not isinstance(cbar, torch.Tensor):
            cbar = torch.as_tensor(cbar, dtype=torch.float64)
        result = torch.zeros(n, dtype=torch.complex128, device=cbar.device)
        for indexes in product(*[(0, 1) for _ in range(ndim)]):
            cbar_slices = tuple(
                slice(i * s, (i + 1) * s) for i, s in zip(indexes, n)
            )
            res = cbar[cbar_slices]
            padding = tuple((i, 0) for i in indexes)
            result = result + xp.pad(res, padding, mode="constant")

        # --- Vectorized DFT at the requested indices (no FFT) ---
        M = int(np.prod(n))
        result_flat = result.reshape(M)

        # Multi-indices m in [0, n) per dim, shape (M, ndim), row-major.
        grids = [torch.arange(s, dtype=torch.float64, device=cbar.device) for s in n]
        mesh = torch.meshgrid(*grids, indexing="ij")
        m_grid = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # (M, ndim)

        idx_array = torch.as_tensor(indices, dtype=torch.float64)  # (B, ndim)
        # phase[b, l] = exp(-2 pi i * sum_j (k_bj / n_j) * m_lj )
        omega = idx_array / torch.tensor(n, dtype=torch.float64, device=cbar.device)
        exponent = -2j * torch.pi * (omega @ m_grid.T)  # (B, M)
        phase = torch.exp(exponent)

        ep_values = torch.real(phase @ result_flat)
        return ep_values
    
    def __call__(
        self,
        sample: SampleOnRectangularGrid,
        model: CovarianceModel,
        indices: List[tuple[int, ...]]
    ) -> torch.Tensor:
        """
        Compute the stochastic Debiased Whittle likelihood for a batch of frequency indices.
        
        The Whittle likelihood for a batch is:
        L(theta) = (1/|B|) * sum_{k in B} [log(f(k; theta)) + I(k) / f(k; theta)]
        
        where B is the batch of frequency indices, I(k) is the periodogram, and f(k; theta) is the expected periodogram.
        
        Parameters
        ----------
        sample : SampleOnRectangularGrid
            The data sample
        model : CovarianceModel
            The covariance model
        indices : List[tuple[int, ...]]
            List of frequency indices for this batch
            
        Returns
        -------
        torch.Tensor
            The stochastic Whittle likelihood value (scalar)
        """
        # Compute periodogram at the specified indices (using FFT-based implementation)
        p = self.compute_periodogram_at_indices(sample, indices)
        
        # Compute expected periodogram at the specified indices (direct, autograd-compatible)
        ep = self.compute_expected_periodogram_at_indices_autograd(model, indices)
        
        # Compute the Whittle likelihood
        # L = mean(log(ep) + p / ep)
        # Add small constant for numerical stability and to avoid log(0)
        epsilon = 1e-10
        log_ep = torch.log(ep + epsilon)
        ratio = p / (ep + epsilon)
        
        whittle = torch.mean(log_ep + ratio)
        
        return whittle


class SGDTrainer:
    """
    Training algorithm using Stochastic Gradient Descent for Debiased Whittle Likelihood.
    
    Parameters
    ----------
    stochastic_whittle : StochasticDebiasedWhittle
        The stochastic Whittle likelihood object
    model : CovarianceModel
        The covariance model to optimize
    dataset : FrequencyIndexDataset
        The dataset providing frequency indices
    learning_rate : float, optional
        Learning rate for SGD, default is 0.01
    batch_size : int, optional
        Batch size for SGD, default is 32
    n_epochs : int, optional
        Number of training epochs, default is 100
    momentum : float, optional
        Momentum for SGD, default is 0.9
    
    Examples
    --------
    >>> from debiased_spatial_whittle.grids.base import RectangularGrid
    >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
    >>> from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
    >>> from debiased_spatial_whittle.inference.sgd_whittle import FrequencyIndexDataset, StochasticDebiasedWhittle, SGDTrainer
    >>> from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
    >>> 
    >>> # Setup
    >>> grid = RectangularGrid((256, 512))
    >>> true_model = ExponentialModel(rho=torch.tensor(6.0), sigma=torch.tensor(1.0))
    >>> sampler = SamplerOnRectangularGrid(true_model, grid)
    >>> sample = sampler()
    >>> 
    >>> # Create periodogram objects
    >>> periodogram = Periodogram()
    >>> ep = ExpectedPeriodogram(grid, periodogram)
    >>> 
    >>> # Create stochastic Whittle
    >>> stochastic_whittle = StochasticDebiasedWhittle(grid, periodogram, ep)
    >>> 
    >>> # Create dataset
    >>> dataset = FrequencyIndexDataset(grid.n)
    >>> 
    >>> # Create and run trainer
    >>> model = ExponentialModel(rho=torch.tensor(3.0, requires_grad=True), sigma=torch.tensor(1.0, requires_grad=True))
    >>> trainer = SGDTrainer(stochastic_whittle, model, dataset)
    >>> trainer.fit(sample, n_epochs=10)
    """
    
    def __init__(
        self,
        stochastic_whittle: StochasticDebiasedWhittle,
        model: CovarianceModel,
        dataset: FrequencyIndexDataset,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        self.stochastic_whittle = stochastic_whittle
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Build a DataLoader over the frequency indices (reshuffled each epoch)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Store the original parameter values and create leaf tensors.
        # Use get_parameter/set_parameter (full names) so this works with both
        # plain models (e.g. ExponentialModel) and ReparameterizedModel wrappers
        # (e.g. LogScaleReparameterizedModel) where the optimized parameters are
        # in a transformed (e.g. log) space.
        self._original_params = {}
        self._leaf_params: list[torch.Tensor] = []
        self._leaf_param_names: list[str] = []
        self._is_reparameterized = isinstance(self.model, ReparameterizedModel)

        for param_name in self.model.free_parameter_names:
            param_value = self.model.get_parameter(param_name)
            if param_value is None:
                continue

            # Extract the parameter name without model prefix (for storage key)
            if '_' in param_name:
                param_name_only = param_name.split('_', 1)[1]
            else:
                param_name_only = param_name

            # Get the current value (in the model's native parameter space)
            if isinstance(param_value, torch.Tensor):
                orig_value = param_value.detach().item()
            else:
                orig_value = float(param_value)

            self._original_params[param_name_only] = orig_value

            # Create a new leaf tensor with requires_grad=True
            new_param = torch.tensor(orig_value, dtype=torch.float64, requires_grad=True)
            self._leaf_params.append(new_param)
            self._leaf_param_names.append(param_name)

            # Set the parameter on the model via set_parameter so the leaf is
            # wired into the model's compute path. For ReparameterizedModel this
            # stores exp(leaf) on the base model; for plain models it stores the
            # leaf directly.
            self.model.set_parameter(param_name, new_param)
        
        # Create optimizer
        self.optimizer = None
        self._create_optimizer()
    
    def _create_optimizer(self):
        """Create the SGD optimizer with all trainable parameters."""
        # Use the leaf tensors stored in __init__ (not get_parameter, which
        # returns non-leaf tensors for ReparameterizedModel).
        params = [p for p in self._leaf_params if p.is_leaf and p.requires_grad]

        if params:
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )
        else:
            raise ValueError("No trainable leaf parameters found in the model")
    
    def _refresh_parameters(self):
        """Re-set the leaf tensors on the model.

        For ReparameterizedModel, set_parameter stores a derived (non-leaf)
        tensor on the base model; after an optimizer step updates the leaf
        in-place, that derived tensor is stale. Calling set_parameter again
        rebuilds it from the updated leaf. For plain models this is a no-op
        (the leaf is already the model's parameter).
        """
        for name, leaf in zip(self._leaf_param_names, self._leaf_params):
            self.model.set_parameter(name, leaf)

    def fit(
        self,
        sample: SampleOnRectangularGrid,
        n_epochs: int = None,
        verbose: bool = True
    ) -> dict:
        """
        Fit the model to the sample using SGD.

        Each epoch iterates over the full DataLoader of frequency indices,
        performing one SGD update per batch.

        Parameters
        ----------
        sample : SampleOnRectangularGrid
            The data sample
        n_epochs : int, optional
            Number of epochs to train. If None, uses self.n_epochs
        verbose : bool, optional
            Whether to print training progress, default is True

        Returns
        -------
        dict
            Training history with 'loss' key containing the loss values per batch
        """
        if n_epochs is None:
            n_epochs = self.n_epochs

        history = {'loss': []}

        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in self.dataloader:
                # The DataLoader collates tuples of indices into per-dimension
                # tensors; stack them into a (batch_size, ndim) long tensor as
                # expected by the index-based compute methods.
                if isinstance(batch, (tuple, list)):
                    batch_indices = torch.stack(
                        [torch.as_tensor(b, dtype=torch.long) for b in batch], dim=1
                    )
                else:
                    batch_indices = torch.as_tensor(batch, dtype=torch.long)

                # For ReparameterizedModel, refresh the base model's parameters
                # from the (possibly updated) leaf tensors before each forward.
                if self._is_reparameterized:
                    self._refresh_parameters()

                self.optimizer.zero_grad()
                loss = self.stochastic_whittle(sample, self.model, batch_indices)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()
                history['loss'].append(loss_value)
                epoch_losses.append(loss_value)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.6f}")
                # Print current parameter values
                for param_name in self.model.free_parameter_names:
                    param_value = self.model.get_parameter(param_name)
                    if param_value is not None:
                        print(f"  {param_name}: {param_value.item():.6f}")

        return history
    
    def get_parameter_values(self) -> dict:
        """
        Get the current values of all parameters in the model's native
        (optimized) space. For LogScaleReparameterizedModel these are the
        log-space values; use get_actual_parameter_values for the base model
        values.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values
        """
        if self._is_reparameterized:
            self._refresh_parameters()
        return {name: self.model.get_parameter(name) for name in self.model.free_parameter_names}

    def get_actual_parameter_values(self) -> dict:
        """
        Get the current values of all parameters in the base model's space
        (i.e. the actual covariance parameters, not the reparameterized ones).
        For plain models this is identical to get_parameter_values.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their actual values
        """
        if self._is_reparameterized:
            self._refresh_parameters()
            base = self.model.base_model
            return {name: base.get_parameter(name) for name in base.free_parameter_names}
        return self.get_parameter_values()


class SampledPeriodogramsDataset(Dataset):
    """
    Dataset of pre-computed periodograms from multiple samples on the same grid.

    Each of the ``K`` samples is stored together with its parameter values.
    The periodogram of every sample is computed once at construction time.
    ``__getitem__`` returns a single frequency index, the associated parameter
    values, and the periodogram value at that frequency — suitable for
    mini-batch training of a :class:`NnConvolModel` via a debiased Whittle loss.

    Parameters
    ----------
    grid : RectangularGrid
        The common grid on which all samples live.
    samples : list[SampleOnRectangularGrid]
        ``K`` samples on ``grid``.
    param_values : list[tuple]
        ``K`` tuples of parameter values, one per sample.  Each tuple must
        have the same length ``a`` (the number of model parameters), and the
        order must match the declaration order of the ``ModelParameter``
        descriptors on the target :class:`NnConvolModel`.
    periodogram : Periodogram
        Periodogram object used to pre-compute the periodograms.

    Attributes
    ----------
    grid : RectangularGrid
    periodograms : list[torch.Tensor]
        Pre-computed periodograms, one per sample.
    param_values : torch.Tensor
        Stacked parameter values, shape ``(K, a)``.
    n_freq : int
        Total number of frequency points per sample.
    """

    def __init__(
        self,
        grid: RectangularGrid,
        samples: List[SampleOnRectangularGrid],
        param_values: List[tuple],
        periodogram: Periodogram,
    ):
        self.grid = grid
        self.grid_shape = grid.n
        self.n_freq = int(np.prod(grid.n))

        if len(samples) != len(param_values):
            raise ValueError(
                f"Got {len(samples)} samples but {len(param_values)} parameter sets."
            )
        if len(samples) == 0:
            raise ValueError("At least one sample is required.")

        # Pre-compute periodograms (real-valued, shape grid.n)
        self.periodograms = []
        for s in samples:
            p = periodogram(s)
            if not isinstance(p, torch.Tensor):
                p = torch.as_tensor(p, dtype=torch.float64)
            self.periodograms.append(p.real)

        # Stack parameter values into a single tensor (K, a)
        self.param_values = torch.as_tensor(
            param_values, dtype=torch.float64
        )

    def __len__(self) -> int:
        return self.n_freq * len(self.periodograms)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return a single (frequency_index, parameter_values, periodogram_value).

        Parameters
        ----------
        idx : int
            Flat index in ``[0, K * n_freq)``.

        Returns
        -------
        tuple
            ``(freq_index, param_values, periodogram_value)`` where
            ``freq_index`` is a tuple of ``int`` (one per grid dimension),
            ``param_values`` is a ``torch.Tensor`` of shape ``(a,)``, and
            ``periodogram_value`` is a scalar ``torch.Tensor``.
        """
        k = idx // self.n_freq          # which sample
        f = idx % self.n_freq           # which frequency

        # Convert flat frequency index to multi-index
        freq_index = np.unravel_index(f, self.grid_shape)

        # Parameter values for this sample
        params = self.param_values[k]

        # Periodogram value at this frequency
        p_val = self.periodograms[k][freq_index]

        return freq_index, params, p_val


class NnConvolTrainer:
    """
    Trainer for the neural-network weights of a
    :class:`~debiased_spatial_whittle.models.convol.NnConvolModel`.

    The ``ModelParameter`` values (e.g. ``rho``, ``sigma``) are **not**
    optimised — they are set from the dataset as constant inputs to the NN.
    Only the ``torch.nn.Module`` weights of the NnConvolModel are trained.

    The trainer iterates over a :class:`SampledPeriodogramsDataset` via a
    :class:`~torch.utils.data.DataLoader`.  For each batch it:

    1. Groups the batch items by their parameter values (items from the
       same sample share the same parameters).
    2. For each group, sets the model's ``ModelParameter`` values (as
       constants) and computes the expected periodogram at the group's
       frequency indices via the autograd-compatible FFT path.  Gradients
       flow through the NN weights only.
    3. Reassembles the per-group EP values in the original batch order
       (preserving the autograd graph via ``torch.cat`` + indexing).
    4. Computes the debiased Whittle loss ``mean(log(ep) + p / ep)``
       against the batch periodogram values and backpropagates.

    Parameters
    ----------
    stochastic_whittle : StochasticDebiasedWhittle
        The stochastic Whittle likelihood object (provides EP computation).
    model : NnConvolModel
        The NN-based convolutional model to train.
    dataset : SampledPeriodogramsDataset
        Dataset of pre-computed periodograms and parameter values.
    learning_rate : float, optional
        Learning rate for the optimizer, default is 1e-3.
    batch_size : int, optional
        Batch size, default is 256.
    n_epochs : int, optional
        Number of training epochs, default is 100.
    weight_decay : float, optional
        Weight decay (L2 penalty), default is 0.0.
    """

    def __init__(
        self,
        stochastic_whittle: StochasticDebiasedWhittle,
        model,
        dataset: SampledPeriodogramsDataset,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 100,
        weight_decay: float = 0.0,
    ):
        self.stochastic_whittle = stochastic_whittle
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay

        # Use a generator on the same device as the model to avoid
        # device mismatches when torch.set_default_device is set to CUDA.
        device = next(model.nn_module.parameters()).device
        gen = torch.Generator(device=device)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True, generator=gen,
        )

        # Store the model's free parameter names so we can set them from
        # the dataset during training.  These are NOT optimised; they are
        # constant inputs to the NN.
        self._param_names = list(self.model.free_parameter_names)

        # The optimizer only trains the NN module weights.
        trainable = list(self.model.nn_module.parameters())
        if not trainable:
            raise ValueError("No trainable NN parameters found.")
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=self.learning_rate,
            #weight_decay=self.weight_decay,
        )

    def _set_model_params(self, values: torch.Tensor):
        """Set the model's ModelParameter values as non-grad constants.

        Parameters
        ----------
        values : torch.Tensor
            1-D tensor of parameter values, one per ``ModelParameter``.
        """
        for i, name in enumerate(self._param_names):
            self.model.set_parameter(name, values[i].detach())

    def fit(self, verbose: bool = True) -> dict:
        """
        Train the NN weights by iterating over the DataLoader.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print training progress, default is True.

        Returns
        -------
        dict
            Training history with 'loss' key containing per-batch loss values.
        """
        history = {"loss": []}

        for epoch in range(self.n_epochs):
            epoch_losses = []

            for batch in self.dataloader:
                # Unpack the collated batch.
                # freq_indices: tuple of tensors (one per dim), each shape (B,)
                # params: tensor shape (B, a)
                # p_values: tensor shape (B,)
                freq_indices, params, p_values = batch

                # Convert frequency indices to (B, ndim) long tensor
                batch_indices = torch.stack(
                    [torch.as_tensor(fi, dtype=torch.long) for fi in freq_indices],
                    dim=1,
                )

                # Move indices to the model's device (DataLoader collates
                # Python ints into CPU tensors)
                device = next(self.model.nn_module.parameters()).device
                batch_indices = batch_indices.to(device)

                # Group batch items by their parameter values.  Items from
                # the same sample share the same parameters, so each group
                # gets its own EP computation.
                unique_params, inverse_indices = torch.unique(
                    params, dim=0, return_inverse=True
                )

                # Sort items by group so we can concatenate per-group EPs.
                sort_order = torch.argsort(inverse_indices)
                sorted_indices = batch_indices[sort_order]
                sorted_p = p_values[sort_order].to(dtype=torch.float64)

                counts = torch.bincount(
                    inverse_indices, minlength=len(unique_params)
                )
                offsets = torch.cumsum(
                    torch.cat([torch.tensor([0]), counts[:-1]]), dim=0
                )

                self.optimizer.zero_grad()

                ep_parts = []
                for g in range(len(unique_params)):
                    start = offsets[g].item()
                    end = start + counts[g].item()
                    group_indices = sorted_indices[start:end]

                    self._set_model_params(unique_params[g])

                    group_ep = (
                        self.stochastic_whittle
                        .compute_expected_periodogram_at_indices_autograd(
                            self.model, group_indices
                        )
                    )
                    ep_parts.append(group_ep)

                # Concatenate per-group EPs (preserves autograd graph).
                ep_sorted = torch.cat(ep_parts)

                # Compute the loss in sorted order (ep_sorted matches sorted_p).
                epsilon = 1e-10
                loss = torch.mean(
                    torch.log(ep_sorted + epsilon) + sorted_p / (ep_sorted + epsilon)
                )

                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()
                history["loss"].append(loss_value)
                epoch_losses.append(loss_value)

            if verbose and (epoch % 2 == 0 or epoch == self.n_epochs - 1):
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {mean_loss:.6f}")

        return history
