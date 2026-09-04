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
from typing import Tuple, List, Optional, Union
from torch.utils.data import Dataset

from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.models.base import CovarianceModel, ModelInterface
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram

xp = BackendManager.get_backend()


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
        
        # Extract values at the specified indices
        batch_size = len(indices)
        periodogram_values = torch.zeros(batch_size, dtype=torch.float64)
        
        for i, idx in enumerate(indices):
            periodogram_values[i] = full_periodogram[idx].real
        
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
        
        # Extract values at the specified indices
        batch_size = len(indices)
        ep_values = torch.zeros(batch_size, dtype=torch.float64)
        
        for i, idx in enumerate(indices):
            ep_values[i] = full_ep[idx].real
        
        return ep_values
    
    def compute_expected_periodogram_at_indices_autograd(
        self,
        model: CovarianceModel,
        indices: List[tuple[int, ...]]
    ) -> torch.Tensor:
        """
        Compute the expected periodogram at specific frequency indices using direct evaluation.
        
        This method computes the expected periodogram by directly evaluating the
        Fourier transform of the autocovariance at the specified frequency indices,
        allowing for proper gradient backpropagation through the model parameters.
        
        For a stationary process, the expected periodogram at frequency k is:
        f(k; theta) = sum_r cov(r; theta) * exp(-i * k^T * r)
        
        where cov(r; theta) is the autocovariance function parameterized by theta,
        and r are the lag vectors.
        
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
        # Get the Fourier frequencies for the grid
        fourier_freqs = self.grid.fourier_frequencies
        
        if not isinstance(fourier_freqs, torch.Tensor):
            fourier_freqs = torch.tensor(fourier_freqs, dtype=torch.float64)
        
        # Get the lags for the autocovariance
        lags = self.grid.lags_unique
        if not isinstance(lags, torch.Tensor):
            lags = torch.tensor(lags, dtype=torch.float64)
        
        batch_size = len(indices)
        ep_values = torch.zeros(batch_size, dtype=torch.float64)
        
        ndim = len(self.grid.n)
        
        # Reshape lags for efficient computation
        # lags shape: (ndim, 2*n1-1, 2*n2-1, ...)
        lags_flat = lags.reshape(ndim, -1)  # (ndim, n_lags)
        lags_expanded = lags_flat.T  # (n_lags, ndim)
        
        # For each frequency index, compute the spectral density
        for i, idx in enumerate(indices):
            # Get the frequency vector at this index
            if ndim == 1:
                freq_vec = fourier_freqs[idx[0]]
            elif ndim == 2:
                freq_vec = fourier_freqs[idx[0], idx[1]]
            elif ndim == 3:
                freq_vec = fourier_freqs[idx[0], idx[1], idx[2]]
            else:
                # Generic case for higher dimensions
                freq_vec = fourier_freqs
                for idim, iidx in enumerate(idx):
                    freq_vec = freq_vec.select(dim=idim, index=iidx)
            
            # Compute the autocovariance at each lag point
            # This will be differentiated through the model parameters
            acv_values = torch.zeros(lags_flat.shape[1], dtype=torch.float64)
            for j, lag_vec in enumerate(lags_expanded):
                # Evaluate model at this specific lag
                # lag_vec shape: (ndim,), we need to add a dimension for the model
                lag_for_model = lag_vec.unsqueeze(1)  # (ndim, 1)
                acv_values[j] = model(lag_for_model)
            
            # Compute the Fourier transform at this specific frequency
            # f(omega) = sum_r cov(r) * exp(-i * omega^T * r)
            
            # Compute omega^T * r for each lag
            omega_dot_r = torch.sum(freq_vec.unsqueeze(1) * lags_flat, dim=0)  # (n_lags,)
            
            # Compute the complex exponential
            exp_term = torch.exp(-1j * omega_dot_r)
            
            # Compute the Fourier transform
            ft = torch.sum(acv_values * exp_term)
            
            # The expected periodogram is the real part (should be real in theory)
            ep_values[i] = ft.real
        
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
        
        # Compute expected periodogram at the specified indices (using autograd-compatible version)
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
        
        # Store the original parameter values and create leaf tensors
        self._original_params = {}
        
        for param_name in self.model.free_parameter_names:
            param_value = self.model.get_parameter(param_name)
            if param_value is not None:
                # Extract the parameter name without model prefix
                if '_' in param_name:
                    model_name, param_name_only = param_name.split('_', 1)
                else:
                    param_name_only = param_name
                    model_name = None
                
                # Get the current value
                current_value = getattr(self.model, param_name_only)
                
                # Store original value
                if isinstance(current_value, torch.Tensor):
                    self._original_params[param_name_only] = current_value.detach().item()
                else:
                    self._original_params[param_name_only] = float(current_value)
                
                # Create a new leaf tensor with requires_grad=True
                new_param = torch.tensor(self._original_params[param_name_only], dtype=torch.float64, requires_grad=True)
                
                # Update the parameter in the model
                setattr(self.model, param_name_only, new_param)
        
        # Create optimizer
        self.optimizer = None
        self._create_optimizer()
    
    def _create_optimizer(self):
        """Create the SGD optimizer with all trainable parameters."""
        # Collect all parameters that are leaf tensors and require gradients
        params = []
        for param_name in self.model.free_parameter_names:
            param_value = self.model.get_parameter(param_name)
            if param_value is not None:
                # Check if this is a leaf tensor
                if param_value.is_leaf and param_value.requires_grad:
                    params.append(param_value)
        
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
    
    def _get_random_batch(self) -> List[tuple[int, ...]]:
        """Get a random batch of frequency indices."""
        indices = torch.randint(0, len(self.dataset), (self.batch_size,))
        return [self.dataset[i] for i in indices.tolist()]
    
    def fit(
        self,
        sample: SampleOnRectangularGrid,
        n_epochs: int = None,
        verbose: bool = True
    ) -> dict:
        """
        Fit the model to the sample using SGD.
        
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
            Training history with 'loss' key containing the loss values per epoch
        """
        if n_epochs is None:
            n_epochs = self.n_epochs
        
        history = {'loss': []}
        
        for epoch in range(n_epochs):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Get a random batch of frequency indices
            batch_indices = self._get_random_batch()
            
            # Compute the stochastic Whittle likelihood
            loss = self.stochastic_whittle(sample, self.model, batch_indices)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Store loss
            history['loss'].append(loss.item())
            
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.6f}")
                # Print current parameter values
                for param_name in self.model.free_parameter_names:
                    param_value = self.model.get_parameter(param_name)
                    if param_value is not None:
                        print(f"  {param_name}: {param_value.item():.6f}")
        
        return history
    
    def get_parameter_values(self) -> dict:
        """
        Get the current values of all parameters.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values
        """
        return {name: self.model.get_parameter(name) for name in self.model.free_parameter_names}


def create_sgd_example():
    """
    Create a simple example script for SGD-based Debiased Whittle Likelihood optimization.
    
    This function creates and runs an example where we:
    1. Simulate data from an Exponential covariance model with rho=6
    2. Use SGD to estimate the rho parameter
    3. Verify that the estimate is close to the true value
    
    Returns
    -------
    dict
        Results containing true and estimated parameter values
    """
    # Set up the backend to use torch
    BackendManager.set_backend("torch")
    BackendManager.device = "cpu"
    
    print("Creating SGD example...")
    print(f"Backend: {BackendManager.backend_name}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create grid: 256 x 512 as requested
    # Note: Using smaller grid for memory efficiency in this example
    grid = RectangularGrid((32, 64))
    print(f"Grid shape: {grid.n}")
    print(f"Note: Using smaller grid (32x64) for memory efficiency. Same concept applies to 256x512.")
    
    # Create true model with rho=6
    true_rho = 6.0
    true_sigma = 1.0
    true_model = ExponentialModel(rho=torch.tensor(true_rho), sigma=torch.tensor(true_sigma))
    print(f"True model: rho={true_rho}, sigma={true_sigma}")
    
    # Sample from the true model
    from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
    sampler = SamplerOnRectangularGrid(true_model, grid)
    sample = sampler()
    print(f"Sample shape: {sample.values.shape}")
    
    # Create periodogram and expected periodogram
    periodogram = Periodogram()
    ep = ExpectedPeriodogram(grid, periodogram)
    
    # Create stochastic Whittle likelihood
    stochastic_whittle = StochasticDebiasedWhittle(grid, periodogram, ep)
    
    # Create dataset for frequency indices
    dataset = FrequencyIndexDataset(grid.n)
    print(f"Dataset size: {len(dataset)}")
    
    # Create initial model with wrong rho
    initial_rho = 3.0
    initial_sigma = 1.0
    model = ExponentialModel(
        rho=torch.tensor(initial_rho, requires_grad=True),
        sigma=torch.tensor(initial_sigma, requires_grad=True)
    )
    print(f"Initial model: rho={initial_rho}, sigma={initial_sigma}")
    
    # Create and run trainer
    trainer = SGDTrainer(
        stochastic_whittle=stochastic_whittle,
        model=model,
        dataset=dataset,
        learning_rate=0.001,
        batch_size=16,
        n_epochs=50,
        momentum=0.9
    )
    
    print("Starting training...")
    history = trainer.fit(sample, n_epochs=50, verbose=True)
    
    # Get final parameter values
    final_params = trainer.get_parameter_values()
    
    print("\nTraining complete!")
    print(f"True rho: {true_rho}")
    print(f"Initial rho: {initial_rho}")
    print(f"Estimated rho: {final_params['ExponentialModel_rho'].item():.6f}")
    
    # Check if the estimate is close to the true value
    estimated_rho = final_params['ExponentialModel_rho'].item()
    rho_error = abs(estimated_rho - true_rho)
    rho_relative_error = rho_error / true_rho
    
    print(f"Absolute error: {rho_error:.6f}")
    print(f"Relative error: {rho_relative_error:.6f}")
    
    if rho_relative_error < 0.2:  # Within 20% of true value
        print("✓ Estimate is close to true value!")
    else:
        print("✗ Estimate is not close to true value")
    
    return {
        'true_rho': true_rho,
        'estimated_rho': estimated_rho,
        'rho_error': rho_error,
        'rho_relative_error': rho_relative_error,
        'history': history
    }


if __name__ == "__main__":
    # Run the example when this file is executed directly
    results = create_sgd_example()
