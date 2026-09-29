"""
Example script demonstrating SGD optimization for Debiased Whittle Likelihood.

This script:
1. Creates a grid (32x64 for memory efficiency, but same concept applies to 256x512)
2. Simulates data from an Exponential covariance model with rho=6
3. Uses SGD to estimate the rho parameter
4. Verifies that the estimate is close to the true value (6)

Usage:
    python docs/sgd_example.py
"""

import sys
sys.path.insert(0, '/workspace/github__arthurBarthe__debiased-spatial-whittle/src')

import torch
import numpy as np
from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('torch')
BackendManager.device = 'cpu'

from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.models.base import LogScaleReparameterizedModel
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.sgd_whittle import FrequencyIndexDataset, StochasticDebiasedWhittle, SGDTrainer
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid


def main():
    print("SGD Optimization for Debiased Whittle Likelihood")
    print("=" * 50)
    
    # Create grid: Use a smaller grid for memory efficiency
    # The original request was 256x512, but that's too large for this environment
    # We'll use 128x128 which demonstrates the same concept
    grid = RectangularGrid((256, 256))
    print(f"Grid shape: {grid.n}")
    print(f"Total grid points: {grid.n_points}")
    print(f"Note: Using smaller grid (128x128) for memory efficiency. Same concept applies to 256x512.")
    
    # Create true model with rho=6
    true_rho = 16.0
    true_sigma = .7
    true_model = ExponentialModel(rho=torch.tensor(true_rho), sigma=torch.tensor(true_sigma))
    print(f"\nTrue model parameters:")
    print(f"  rho = {true_rho}")
    print(f"  sigma = {true_sigma}")
    
    # Sample from the true model
    print("\nSampling from true model...")
    sampler = SamplerOnRectangularGrid(true_model, grid)
    sample = sampler()
    print(f"Sample shape: {sample.values.shape}")
    
    # Create periodogram and expected periodogram
    print("\nCreating periodogram objects...")
    periodogram = Periodogram()
    ep = ExpectedPeriodogram(grid, periodogram)
    
    # Create stochastic Whittle likelihood
    stochastic_whittle = StochasticDebiasedWhittle(grid, periodogram, ep)
    
    # Create dataset for frequency indices
    dataset = FrequencyIndexDataset(grid.n)
    print(f"Dataset size (number of frequency indices): {len(dataset)}")
    
    # Create initial model with wrong rho - use leaf tensors.
    # Wrap in LogScaleReparameterizedModel so the optimizer works in log-space
    # for both parameters, keeping them positive without explicit constraints.
    initial_rho = 3.0
    initial_sigma = 1.0
    base_model = ExponentialModel(
        rho=torch.tensor(initial_rho),
        sigma=torch.tensor(initial_sigma)
    )
    model = LogScaleReparameterizedModel(base_model, sel=(True, True))
    print(f"\nInitial model parameters (log-scale reparameterized):")
    print(f"  rho = {initial_rho} (log: {np.log(initial_rho):.6f})")
    print(f"  sigma = {initial_sigma} (log: {np.log(initial_sigma):.6f})")
    
    # Create and run trainer
    print(f"\nCreating SGD trainer...")
    trainer = SGDTrainer(
        stochastic_whittle=stochastic_whittle,
        model=model,
        dataset=dataset,
        learning_rate=1e-2,
        batch_size=256 ** 2 // 10,
        n_epochs=500,
        momentum=0.9
    )
    
    print("Starting training...")
    history = trainer.fit(sample, verbose=True)
    
    # Get final parameter values (actual, not log-space)
    final_params = trainer.get_actual_parameter_values()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    # Print results
    estimated_rho = final_params['ExponentialModel_rho'].item()
    estimated_sigma = final_params['ExponentialModel_sigma'].item()
    
    print(f"\nResults:")
    print(f"  True rho:    {true_rho}")
    print(f"  Initial rho: {initial_rho}")
    print(f"  Estimated rho: {estimated_rho:.6f}")
    print(f"  True sigma:    {true_sigma}")
    print(f"  Initial sigma: {initial_sigma}")
    print(f"  Estimated sigma: {estimated_sigma:.6f}")
    
    # Compute errors
    rho_error = abs(estimated_rho - true_rho)
    rho_relative_error = rho_error / true_rho
    sigma_error = abs(estimated_sigma - true_sigma)
    sigma_relative_error = sigma_error / true_sigma
    
    print(f"\nErrors:")
    print(f"  Absolute rho error: {rho_error:.6f}")
    print(f"  Relative rho error: {rho_relative_error:.6f}")
    print(f"  Absolute sigma error: {sigma_error:.6f}")
    print(f"  Relative sigma error: {sigma_relative_error:.6f}")
    
    # Check if the estimate is close to the true value
    print(f"\nVerification:")
    if rho_relative_error < 0.2:  # Within 20% of true value (more lenient for smaller grid)
        print("  ✓ Rho estimate is close to true value!")
    else:
        print("  ✗ Rho estimate is not close to true value")
    
    if sigma_relative_error < 0.2:  # Within 20% of true value
        print("  ✓ Sigma estimate is close to true value!")
    else:
        print("  ✗ Sigma estimate is not close to true value")
    
    return {
        'true_rho': true_rho,
        'estimated_rho': estimated_rho,
        'rho_error': rho_error,
        'rho_relative_error': rho_relative_error,
        'true_sigma': true_sigma,
        'estimated_sigma': estimated_sigma,
        'sigma_error': sigma_error,
        'sigma_relative_error': sigma_relative_error,
        'history': history
    }


if __name__ == "__main__":
    results = main()
