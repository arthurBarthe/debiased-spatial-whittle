"""
Example: training a neural network to directly learn the tapered expected
periodogram of Exponential covariance models in 3D via the debiased Whittle
likelihood.

This script:
1. Generates multiple samples from ExponentialModel with various rho values
   (sigma = 1) on a 32x32x32 grid and computes their tapered periodograms
   (Hanning taper) with zero-padding to (2*n1, 2*n2, 2*n3).
2. Builds a plain MLP that takes (scaled_log_rho, sigma, f1, f2, f3) as input
   and outputs ln(EP) directly.  Trained via the debiased Whittle loss.
3. Plots histograms of residuals 1 - exp(-P/EP) for each rho value.
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
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.inference.tapers import HanningTaper
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.nn_ep import (
    InverseRadiusFrequencySampler,
    PaddedPeriodogramDataset,
    NnEpTrainer,
)
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid

device = "cuda:0"


def main():
    # --- Grid ---
    grid = RectangularGrid((32, 32, 32))
    d = grid.ndim
    padded_n = tuple(2 * s for s in grid.n)
    N_freq = int(np.prod(padded_n))
    print(f"Grid: {grid.n}, {d}D, padded: {padded_n}")

    # --- True parameter sets ---
    rho_values = [2.0, 4.0, 8.0, 16.0]
    sigma_value = 1.0
    K = len(rho_values)
    n_samples_per_rho = 2

    # Rescale log(rho) to [-1, 1] over the training range
    log_rho_values = [np.log(rho) for rho in rho_values]
    log_rho_min = min(log_rho_values)
    log_rho_max = max(log_rho_values)
    def scale_log_rho(log_rho):
        return 2.0 * (log_rho - log_rho_min) / (log_rho_max - log_rho_min) - 1.0

    # --- Tapered periodogram with fold=False ---
    taper = HanningTaper()
    periodogram = Periodogram(taper=taper)
    periodogram.fold = False
    ep = ExpectedPeriodogram(grid, periodogram)

    # --- Generate samples ---
    print("Generating samples...")
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
    print(f"  {len(samples)} samples ({n_samples_per_rho} per rho)")

    # --- Build frequency grid ---
    freq_1d = [torch.fft.fftfreq(2 * n_i, d_i) for n_i, d_i in zip(grid.n, grid.delta)]
    freq_mesh = torch.meshgrid(*freq_1d, indexing="ij")
    freq_grid = torch.stack(freq_mesh, axis=-1)  # (2*n1, 2*n2, 2*n3, d)

    # --- Build dataset ---
    print("Building dataset...")
    freq_fraction = 0.05
    n_freq_total = int(np.prod(padded_n))
    n_freq_per_sample = max(1, int(n_freq_total * freq_fraction))
    freq_sampler = InverseRadiusFrequencySampler(
        n_freq_total, n_freq_per_sample, padded_n, eps=0.01
    )
    dataset = PaddedPeriodogramDataset(
        grid, samples, param_values, periodogram,
        freq_fraction=freq_fraction, freq_sampler=freq_sampler,
    )
    print(f"  {dataset.n_samples} samples, {dataset.n_freq_per_sample} freqs/sample "
          f"({freq_fraction*100:.0f}% of {dataset.n_freq_total})")

    # --- Build the neural network ---
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
    ).to(device)

    # --- Train ---
    trainer = NnEpTrainer(
        nn_module=nn_module,
        dataset=dataset,
        freq_grid=freq_grid,
        device=device,
        learning_rate=1e-2,
        batch_size=8192,
        n_epochs=60,
    )
    print("Training (Whittle loss, fold=False, 3D)...")
    trainer.fit(verbose=True)

    # --- Save the trained model ---
    model_path = "nn_spectral_ep_model_3d.pt"
    torch.save({
        "state_dict": nn_module.state_dict(),
        "architecture": {
            "in_dim": a + d,
            "hidden_dims": [128, 256, 256, 128],
            "out_dim": 1,
        },
        "metadata": {
            "grid_n": grid.n,
            "ndim": d,
            "n_params": a,
            "rho_values": rho_values,
            "sigma_value": sigma_value,
            "log_rho_min": log_rho_min,
            "log_rho_max": log_rho_max,
        },
    }, model_path)
    print(f"Model saved to {model_path}")

    # --- Diagnostic plots: residual histograms ---
    print("Computing residuals for plotting...")
    nn_module.eval()

    diag_hists = []

    for col, rho in enumerate(rho_values):
        # Trained EP
        scaled = scale_log_rho(np.log(rho))
        params = torch.tensor([scaled, sigma_value], dtype=torch.float64)
        pred_ep = trainer.predict_ep(params)

        # Diagnostic: 1 - exp(-p / ep) using the first sample's periodogram
        p_vals = dataset.periodograms[col * n_samples_per_rho].cpu()
        diag = 1.0 - torch.exp(-p_vals / pred_ep)
        diag_hists.append(diag.numpy().flatten())

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    if K == 1:
        axes = [axes]

    for col, rho in enumerate(rho_values):
        axes[col].hist(diag_hists[col], bins=100, density=True,
                       color="steelblue", edgecolor="none")
        axes[col].set_title(f"1 - exp(-P/EP) (rho={rho})")
        axes[col].set_xlabel("1 - exp(-P/EP)")
        axes[col].set_ylabel("Density")

    fig.suptitle("3D Tapered EP: Residual histograms (Whittle loss)", fontsize=14)
    fig.tight_layout()
    plt.savefig("nn_spectral_example_3d.png", dpi=150, bbox_inches="tight")
    print("Figure saved to nn_spectral_example_3d.png")


if __name__ == "__main__":
    main()
