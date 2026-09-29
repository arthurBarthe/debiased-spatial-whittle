"""
Example: training a neural network to directly learn the tapered expected
periodogram of Matern covariance models via the debiased Whittle likelihood.

This script:
1. Generates multiple samples from Matern covariance models with various
   (rho, nu) values (sigma = 1) on a 256x256 grid and computes their tapered
   periodograms (Hanning taper) with zero-padding to (2*n1, 2*n2). The Matern
   smoothness nu is taken from {0.5 (ExponentialModel), 1.5 (Matern32),
   2.5 (Matern52)} and rho from {2, 4, 8, 16, 32}.
2. Builds a plain MLP that takes (scaled_log_rho, scaled_nu, f1, f2) as input
   and outputs ln(EP) directly.  The NN-predicted EP is used inside the
   debiased Whittle loss:  mean( ln(EP) + P / EP )  where P is the tapered
   periodogram. Both rho and nu are treated as continuous inputs to the NN.
3. Plots the true versus trained expected periodograms (log10 scale) for each
   (rho, nu) value, plus the pointwise absolute error and a histogram of
   residuals.

Uses fold=False so the periodogram and expected periodogram have shape
(2*n1, 2*n2) instead of (n1, n2).
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
from debiased_spatial_whittle.models.univariate import (
    ExponentialModel,
    Matern32Model,
    Matern52Model,
)
from debiased_spatial_whittle.inference.tapers import HanningTaper
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.nn_ep import (
    InverseRadiusFrequencySampler,
    PaddedPeriodogramDataset,
    NnEpTrainer, UniformFrequencySampler,
)
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid

device = "cuda:0"

# Matern smoothness nu -> model class. Only these discrete nu values have a
# dedicated covariance model class available for sampling / true EP.
NU_MODELS = {
    0.5: ExponentialModel,
    1.5: Matern32Model,
    2.5: Matern52Model,
}
NU_NAMES = {0.5: "Exp", 1.5: "Mat32", 2.5: "Mat52"}


def make_model(rho, nu, sigma=1.0):
    """Build the Matern covariance model matching the nearest discrete nu."""
    nearest = min(NU_MODELS, key=lambda k: abs(k - float(nu)))
    model_cls = NU_MODELS[nearest]
    return model_cls(rho=torch.tensor(rho), sigma=torch.tensor(sigma))


def main():
    # --- Grid ---
    grid = RectangularGrid((128, 128))
    d = grid.ndim
    padded_n = tuple(2 * s for s in grid.n)
    N_freq = int(np.prod(padded_n))
    print(f"Grid: {grid.n}, {d}D, padded: {padded_n}")

    # --- True parameter sets ---
    rho_values = [2.0, 8.0, 16.0, 32]
    nu_values = [0.5, 1.5, 2.5]
    sigma_value = 1.0
    n_samples_per_combo = 2

    # Rescale log(rho) to [-1, 1] over the training range
    log_rho_values = [np.log(rho) for rho in rho_values]
    log_rho_min = min(log_rho_values)
    log_rho_max = max(log_rho_values)
    def scale_log_rho(log_rho):
        return 2.0 * (log_rho - log_rho_min) / (log_rho_max - log_rho_min) - 1.0

    # Rescale nu to [-1, 1] over [0.5, 2.5]
    nu_min, nu_max = 0.5, 2.5
    def scale_nu(nu):
        return 2.0 * (nu - nu_min) / (nu_max - nu_min) - 1.0

    # --- Tapered periodogram with fold=False ---
    taper = HanningTaper()
    periodogram = Periodogram(taper=taper)
    periodogram.fold = False
    ep = ExpectedPeriodogram(grid, periodogram)

    # --- Generate samples ---
    print("Generating samples...")
    samples = []
    param_values = []
    combos = [(rho, nu) for nu in nu_values for rho in rho_values]
    for rho, nu in combos:
        scaled_rho = scale_log_rho(np.log(rho))
        scaled_nu = scale_nu(nu)
        for _ in range(n_samples_per_combo):
            model = make_model(rho, nu, sigma=sigma_value)
            sampler = SamplerOnRectangularGrid(model, grid)
            sample = sampler()
            samples.append(sample)
            param_values.append((scaled_rho, scaled_nu))
    K = len(combos)
    print(f"  {len(samples)} samples ({n_samples_per_combo} per (rho, nu))")

    # --- Build frequency grid ---
    freq_1d = [torch.fft.fftfreq(2 * n_i, d_i) for n_i, d_i in zip(grid.n, grid.delta)]
    freq_mesh = torch.meshgrid(*freq_1d, indexing="ij")
    freq_grid = torch.stack(freq_mesh, axis=-1)  # (2*n1, 2*n2, d)

    # --- Build dataset ---
    print("Building dataset...")
    freq_fraction = 0.1
    n_freq_total = int(np.prod(padded_n))
    n_freq_per_sample = max(1, int(n_freq_total * freq_fraction))
    freq_sampler = InverseRadiusFrequencySampler(
        n_freq_total, n_freq_per_sample, padded_n, eps=0.001
    )
    dataset = PaddedPeriodogramDataset(
        grid, samples, param_values, periodogram,
        freq_fraction=freq_fraction, freq_sampler=freq_sampler,
    )
    print(f"  {dataset.n_samples} samples, {dataset.n_freq_per_sample} freqs/sample "
          f"({freq_fraction*100:.0f}% of {dataset.n_freq_total})")

    # --- Build the neural network ---
    a = 2  # scaled_log_rho, scaled_nu
    nn_module = torch.nn.Sequential(
        torch.nn.Linear(a + d, 256, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1, dtype=torch.float64),
    ).to(device)

    # --- Train ---
    trainer = NnEpTrainer(
        nn_module=nn_module,
        dataset=dataset,
        freq_grid=freq_grid,
        device=device,
        learning_rate=1e-2,
        batch_size=4096,
        n_epochs=40,
    )
    print("Training (Whittle loss, fold=False)...")
    trainer.fit(verbose=True)

    # --- Save the trained model ---
    model_path = "nn_spectral_ep_model.pt"
    torch.save({
        "state_dict": nn_module.state_dict(),
        "architecture": {
            "in_dim": a + d,
            "hidden_dims": [256, 512, 512, 256],
            "out_dim": 1,
        },
        "metadata": {
            "grid_n": grid.n,
            "ndim": d,
            "n_params": a,
            "rho_values": rho_values,
            "nu_values": nu_values,
            "sigma_value": sigma_value,
            "log_rho_min": log_rho_min,
            "log_rho_max": log_rho_max,
            "nu_min": nu_min,
            "nu_max": nu_max,
        },
    }, model_path)
    print(f"Model saved to {model_path}")

    # --- Plot true vs trained EP ---
    print("Computing true and trained EPs for plotting...")
    nn_module.eval()

    true_ep_logs = []
    trained_ep_logs = []
    err_maps = []
    diag_hists = []

    for col, (rho, nu) in enumerate(combos):
        # True EP
        true_model = make_model(rho, nu, sigma=sigma_value)
        with torch.no_grad():
            true_ep = ep(true_model)
        true_ep_log = torch.log10(torch.maximum(true_ep.real, torch.tensor(1e-12)))

        # Trained EP via trainer.predict_ep
        scaled_rho = scale_log_rho(np.log(rho))
        scaled_nu = scale_nu(nu)
        params = torch.tensor([scaled_rho, scaled_nu], dtype=torch.float64)
        pred_ep = trainer.predict_ep(params)
        pred_log10 = torch.log10(pred_ep)

        # Diagnostic: 1 - exp(-p / ep)
        p_vals = dataset.periodograms[col * n_samples_per_combo].cpu()
        diag = 1.0 - torch.exp(-p_vals / pred_ep)
        diag_hists.append(diag.numpy().flatten())

        true_np = np.fft.fftshift(true_ep_log.cpu().numpy())
        trained_np = np.fft.fftshift(pred_log10.cpu().numpy())
        err_np = np.abs(trained_np - true_np)

        true_ep_logs.append(true_np)
        trained_ep_logs.append(trained_np)
        err_maps.append(err_np)

    ep_vmin = min(min(t.min() for t in true_ep_logs),
                  min(t.min() for t in trained_ep_logs))
    ep_vmax = max(max(t.max() for t in true_ep_logs),
                  max(t.max() for t in trained_ep_logs))
    err_vmax = max(e.max() for e in err_maps)

    fig, axes = plt.subplots(4, K, figsize=(2.5 * K, 13))

    for col, (rho, nu) in enumerate(combos):
        title = f"rho={rho:g}, nu={nu:g} ({NU_NAMES[nu]})"

        im1 = axes[0, col].imshow(
            true_ep_logs[col], origin="lower", cmap="viridis",
            vmin=ep_vmin, vmax=ep_vmax,
        )
        axes[0, col].set_title(f"True EP ({title})")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        im2 = axes[1, col].imshow(
            trained_ep_logs[col], origin="lower", cmap="viridis",
            vmin=ep_vmin, vmax=ep_vmax,
        )
        axes[1, col].set_title(f"Trained EP ({title})")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        im3 = axes[2, col].imshow(
            err_maps[col], origin="lower", cmap="hot",
            vmin=0, vmax=err_vmax,
        )
        axes[2, col].set_title(f"|Error| ({title})")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        axes[3, col].hist(diag_hists[col], bins=100, density=True,
                          color="steelblue", edgecolor="none")
        axes[3, col].set_title(f"1 - exp(-P/EP) ({title})")
        axes[3, col].set_xlabel("1 - exp(-P/EP)")

    axes[0, 0].set_ylabel("True log10(EP)", fontsize=12)
    axes[1, 0].set_ylabel("Trained log10(EP)", fontsize=12)
    axes[2, 0].set_ylabel("|Trained - True|", fontsize=12)
    axes[3, 0].set_ylabel("Density", fontsize=12)

    fig.suptitle("Tapered EP (fold=False): True vs Trained NN (Matern, Whittle loss, log10 scale)", fontsize=14)
    fig.tight_layout()
    fig.colorbar(im2, ax=axes[:2, :], shrink=0.6, label="log10(EP)")
    fig.colorbar(im3, ax=axes[2, :], shrink=0.6, label="|error|")

    plt.savefig("nn_spectral_example.png", dpi=150, bbox_inches="tight")
    print("Figure saved to nn_spectral_example.png")


if __name__ == "__main__":
    main()
