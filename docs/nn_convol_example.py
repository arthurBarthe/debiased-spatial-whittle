"""
Example: training a NnConvolModel to match Exponential covariance models.

This script:
1. Creates 4 samples per rho from an ExponentialModel with rho = 2, 4, 8, 16
   (sigma = 1) on a 64x64 grid.  Multiple samples per rho reduce the variance
   of the empirical periodogram, giving a cleaner signal for the Whittle loss.
2. Builds a NnConvolModel with an MLP that takes (scaled_log_rho, sigma, i1, i2)
   as input, where scaled_log_rho is log(rho) rescaled to [-1, 1] over the
   training range.  The sample is mean-subtracted, variance-normalized, and
   scaled by sigma — this decouples the amplitude from the spatial correlation
   structure so the NN can focus on learning the correlation shape.
3. Trains the NN weights via a debiased Whittle loss using a DataLoader over
   a SampledPeriodogramsDataset, with a 3-phase learning rate schedule
   (1e-2, 1e-3, 1e-4).
4. Plots the expected periodogram for the true ExponentialModel (first row)
   and the trained NnConvolModel (second row), plus pre- and post-training samples,
   for each rho value.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("torch")
BackendManager.device = "cuda:0"

from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.convol import NnConvolModel
from debiased_spatial_whittle.models.base import ModelParameter
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.inference.tapers import HanningTaper
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.sgd_whittle import (
    StochasticDebiasedWhittle,
    SampledPeriodogramsDataset,
    NnConvolTrainer,
)
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid


class ExponentialNnModel(NnConvolModel):
    """NnConvolModel with scaled_log_rho and sigma parameters, matching ExponentialModel.

    The NN receives log(rho) rescaled to [-1, 1] over the training range as input,
    which gives a balanced dynamic range for the NN.

    The sample is mean-subtracted and variance-normalized to unit variance, then
    scaled by sigma.  This decouples the amplitude (controlled by sigma) from the
    spatial correlation structure (learned by the NN), preventing the optimizer
    from trivially adjusting the field scale instead of learning the correlation
    shape.
    """

    scaled_log_rho = ModelParameter(default=0.0, bounds=(-1.0, 1.0), latex_display=r"\tilde{\log\rho}")
    sigma = ModelParameter(default=1.0, bounds=(0, float("inf")), latex_display=r"\sigma")

    def __init__(self, grid, nn_module, scaled_log_rho=None, sigma=None, name=None):
        super().__init__(grid, nn_module, scaled_log_rho, sigma, name=name)

    def _compute_sample(self):
        """Evaluate the NN at every grid index, then normalize variance."""
        n = self.grid.n
        param_names = self._parameters
        a = len(param_names)

        try:
            nn_param = next(self.nn_module.parameters())
            nn_dtype = nn_param.dtype
            nn_device = nn_param.device
        except StopIteration:
            nn_dtype = torch.float64
            nn_device = None

        param_values = []
        for pname in param_names:
            val = getattr(self, pname)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=torch.float64)
            val = val.to(dtype=nn_dtype, device=nn_device)
            param_values.append(val)

        coords_1d = [
            torch.arange(n_i, dtype=nn_dtype, device=nn_device) / max(n_i - 1, 1) - 0.5
            for n_i in n
        ]
        mesh = torch.meshgrid(*coords_1d, indexing="ij")
        coords_flat = torch.stack([m.reshape(-1) for m in mesh], dim=1)

        N = coords_flat.shape[0]
        if a > 0:
            param_rows = [v.reshape(1).expand(N) for v in param_values]
            params_flat = torch.stack(param_rows, dim=1)
            nn_input = torch.cat([params_flat, coords_flat], dim=1)
        else:
            nn_input = coords_flat

        sample_flat = self.nn_module(nn_input)
        if sample_flat.dim() > 1 and sample_flat.shape[-1] == 1:
            sample_flat = sample_flat.squeeze(-1)

        sample = sample_flat.reshape(*n)
        sample = sample - torch.mean(sample)
        std = torch.std(sample)
        if std > 1e-12:
            sample = sample / std
        return sample


def main():
    # --- Grid ---
    grid = RectangularGrid((32, 32), (2, 2))
    d = grid.ndim
    print(f"Grid: {grid.n}, {d}D")

    # --- True parameter sets ---
    rho_values = [2.0, 4.0, 8.0, 16.0]
    sigma_value = 1.0
    K = len(rho_values)
    n_samples_per_rho = 12

    # Rescale log(rho) to [-1, 1] over the training range
    log_rho_values = [np.log(rho) for rho in rho_values]
    log_rho_min = min(log_rho_values)
    log_rho_max = max(log_rho_values)
    def scale_log_rho(log_rho):
        return 2.0 * (log_rho - log_rho_min) / (log_rho_max - log_rho_min) - 1.0

    # --- Generate samples ---
    print("Generating samples...")
    periodogram = Periodogram(taper=HanningTaper())
    samples = []
    param_values = []
    for rho in rho_values:
        scaled = scale_log_rho(np.log(rho))
        for _ in range(n_samples_per_rho):
            model = ExponentialModel(
                rho=torch.tensor(rho), sigma=torch.tensor(sigma_value)
            )
            sampler = SamplerOnRectangularGrid(model, grid)
            sample = sampler()
            samples.append(sample)
            param_values.append((scaled, sigma_value))
            print(f"  rho={rho}, scaled_log_rho={scaled:.4f}, sigma={sigma_value}")

    # --- Build the neural network ---
    # Plain MLP without BatchNorm.  The variance normalization in _compute_sample
    # handles amplitude control, so BatchNorm is unnecessary and can cause scale
    # instabilities in the autocorrelation.
    a = 2  # scaled_log_rho, sigma
    nn_module = torch.nn.Sequential(
        torch.nn.Linear(a + d, 128, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 256, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1, dtype=torch.float64),
        torch.nn.Tanh()
    ).to("cuda:0")

    # --- Create the NnConvolModel ---
    nn_model = ExponentialNnModel(grid, nn_module, scaled_log_rho=scale_log_rho(np.log(8.0)), sigma=1.0)

    # --- Dataset ---
    print("Building dataset (pre-computing periodograms)...")
    dataset = SampledPeriodogramsDataset(grid, samples, param_values, periodogram)
    print(f"Dataset size: {len(dataset)}")

    # --- Trainer ---
    ep = ExpectedPeriodogram(grid, periodogram)
    stochastic_whittle = StochasticDebiasedWhittle(grid, periodogram, ep)

    trainer = NnConvolTrainer(
        stochastic_whittle=stochastic_whittle,
        model=nn_model,
        dataset=dataset,
        learning_rate=5e-3,
        batch_size=4096,
        n_epochs=20,
    )

    print("Training...")
    # Capture pre-training samples for comparison
    nn_module.eval()
    pre_training_samples = []
    for rho in rho_values:
        nn_model.set_parameter("ExponentialNnModel_scaled_log_rho", torch.tensor(scale_log_rho(np.log(rho))))
        nn_model.set_parameter("ExponentialNnModel_sigma", torch.tensor(sigma_value))
        with torch.no_grad():
            pre_training_samples.append(nn_model._compute_sample().cpu().numpy())

    # Phase 1: high learning rate
    nn_module.train()
    print("Phase 1: LR=1e-2, 20 epochs")
    history = trainer.fit(verbose=True)

    # Phase 2: decay learning rate
    for pg in trainer.optimizer.param_groups:
        pg['lr'] = 1e-3
    trainer.n_epochs = 40
    print("Phase 2: LR=1e-3, 20 epochs")
    history2 = trainer.fit(verbose=True)
    history["loss"].extend(history2["loss"])

    # Phase 3: fine-tune
    for pg in trainer.optimizer.param_groups:
        pg['lr'] = 1e-4
    trainer.n_epochs = 40
    print("Phase 3: LR=1e-4, 20 epochs")
    history3 = trainer.fit(verbose=True)
    history["loss"].extend(history3["loss"])

    # --- Plot expected periodograms ---
    print("Computing expected periodograms for plotting...")
    nn_module.eval()

    fig, axes = plt.subplots(4, K, figsize=(4 * K, 16))

    # Collect all data first so we can use shared vmin/vmax across rho values.
    true_ep_logs = []
    trained_ep_logs = []
    pre_samples = []
    post_samples = []

    for col, rho in enumerate(rho_values):
        # True model EP
        true_model = ExponentialModel(
            rho=torch.tensor(rho), sigma=torch.tensor(sigma_value)
        )
        with torch.no_grad():
            true_ep = ep(true_model)
        true_ep_np = np.fft.fftshift(np.real(true_ep.cpu().numpy()))

        # Trained model EP and sample
        nn_model.set_parameter("ExponentialNnModel_scaled_log_rho", torch.tensor(scale_log_rho(np.log(rho))))
        nn_model.set_parameter("ExponentialNnModel_sigma", torch.tensor(sigma_value))
        print(nn_model)
        with torch.no_grad():
            trained_ep = ep(nn_model)
            trained_sample = nn_model._compute_sample()
        trained_ep_np = np.fft.fftshift(np.real(trained_ep.cpu().numpy()))
        trained_sample_np = trained_sample.cpu().numpy()

        true_ep_logs.append(np.log10(np.maximum(true_ep_np, 1e-12)))
        trained_ep_logs.append(np.log10(np.maximum(trained_ep_np, 1e-12)))
        pre_samples.append(pre_training_samples[col])
        post_samples.append(trained_sample_np)

    # Shared color ranges across all rho values.
    ep_vmin = min(min(t.min() for t in true_ep_logs),
                  min(t.min() for t in trained_ep_logs))
    ep_vmax = max(max(t.max() for t in true_ep_logs),
                  max(t.max() for t in trained_ep_logs))
    sample_vmin = min(min(s.min() for s in pre_samples),
                      min(s.min() for s in post_samples))
    sample_vmax = max(max(s.max() for s in pre_samples),
                      max(s.max() for s in post_samples))

    for col, rho in enumerate(rho_values):
        im1 = axes[0, col].imshow(
            true_ep_logs[col], origin="lower", cmap="viridis",
            vmin=ep_vmin, vmax=ep_vmax,
        )
        axes[0, col].set_title(f"True EP (rho={rho})")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        im2 = axes[1, col].imshow(
            trained_ep_logs[col], origin="lower", cmap="viridis",
            vmin=ep_vmin, vmax=ep_vmax,
        )
        axes[1, col].set_title(f"Trained EP (rho={rho})")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        im3 = axes[2, col].imshow(
            pre_samples[col], origin="lower", cmap="RdBu_r",
            vmin=sample_vmin, vmax=sample_vmax,
        )
        axes[2, col].set_title(f"Pre-training Sample (rho={rho})")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        im4 = axes[3, col].imshow(
            post_samples[col], origin="lower", cmap="RdBu_r",
            vmin=sample_vmin, vmax=sample_vmax,
        )
        axes[3, col].set_title(f"Post-training Sample (rho={rho})")
        axes[3, col].set_xticks([])
        axes[3, col].set_yticks([])

    axes[0, 0].set_ylabel("True Model EP", fontsize=12)
    axes[1, 0].set_ylabel("Trained NN EP", fontsize=12)
    axes[2, 0].set_ylabel("Pre-training Sample", fontsize=12)
    axes[3, 0].set_ylabel("Post-training Sample", fontsize=12)

    fig.suptitle("Expected Periodogram: True vs Trained NnConvolModel", fontsize=14)
    fig.tight_layout()
    fig.colorbar(im2, ax=axes[:2, :], shrink=0.6, label="log10(EP)")
    fig.colorbar(im3, ax=axes[2, :], shrink=0.6, label="sample value")
    fig.colorbar(im4, ax=axes[3, :], shrink=0.6, label="sample value")

    plt.savefig("nn_convol_example.png", dpi=150, bbox_inches="tight")
    print("Figure saved to nn_convol_example.png")


if __name__ == "__main__":
    main()
