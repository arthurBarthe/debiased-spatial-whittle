from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.models.base import BaseCovarianceModel, ModelParameter
from abc import ABCMeta, abstractmethod

xp = BackendManager.get_backend()
fftn, ifftn = BackendManager.get_fft_methods()
arange = BackendManager.get_arange()
gamma = BackendManager.get_gamma()


class SpectralModel(BaseCovarianceModel):
    """
    Base class to define a covariance model from a spectral density function.
    """
    @abstractmethod
    def spectral_density(self, frequencies: xp.ndarray) -> xp.ndarray:
        """
        Abstract method that must provide the spectral density function evaluated at the passed frequencies

        Parameters
        ----------
        frequencies
            shape (n1, ..., nk, d)

        Returns
        -------
        sdf
            shape (n1, ..., nk). Values of the spectral density function
        """
        raise NotImplementedError()

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Compute covariance from spectral density by creating an embedding grid.
        
        Parameters
        ----------
        lags : xp.ndarray
            Array of lag vectors, shape (d, n1, n2, ..., nk) where d is the dimension
        *params : tuple
            Model parameters
            
        Returns
        -------
        xp.ndarray
            Covariance values at the lags, shape (n1, n2, ..., nk)
        """
        from debiased_spatial_whittle.grids.base import RectangularGrid
        
        # lags shape: (d, n1, n2, ..., nk)
        # We need to create a grid that covers all lags
        
        # Get the dimension and the number of lag points
        d = lags.shape[0]  # spatial dimension
        lag_shape = lags.shape[1:]  # shape of lag points (n1, n2, ..., nk)
        
        # Flatten lags to find min and max along each dimension
        # Reshape lags to (d, -1) to process all lag points together
        lags_flat = lags.reshape(d, -1)  # shape (d, N)
        
        # For each dimension, find the min and max lag
        min_lags = xp.min(xp.abs(lags_flat[:, xp.all(lags_flat > 0, 0)]), axis=1)  # shape (d,)
        max_lags = xp.max(lags_flat, axis=1)  # shape (d,)

        grid_delta = min_lags / 2
        grid_n = (max_lags / grid_delta).astype(int)
        
        # Create the embedding grid
        grid = RectangularGrid(tuple(grid_n), delta=tuple(grid_delta))
        
        # Evaluate the model on the grid
        cov_on_grid = self.call_on_rectangular_grid(grid)
        
        # Map lags to grid indices
        indices = []
        for dim in range(d):
            lags_dim = lags[dim]
            indices_dim = xp.round(lags_dim / grid_delta[dim]).astype(int)
            indices.append(indices_dim)
        
        # Extract values from cov_on_grid
        result = cov_on_grid[tuple(indices)]
        
        return result

    def call_on_rectangular_grid(self, grid):
        fftfreq = xp.fft.fftfreq
        ndim = len(grid.n)
        n = grid.n
        delta = grid.delta
        mesh = xp.meshgrid(
            *[fftfreq(3 * n_i + 1, d_i / 2) for n_i, d_i in zip(n, delta)],
            indexing="ij",
        )
        freqs = xp.stack(mesh, axis=-1)  # / (2 * np.pi)
        sdf = self.spectral_density(freqs)
        out = xp.real(fftn(sdf)) / xp.prod(xp.array([(3 * n_i + 1) / 2 for n_i in n]))
        for i_dim in range(ndim):
            n_i = n[i_dim]
            out = xp.take(
                out,
                xp.concatenate(
                    (
                        arange(0, 2 * n_i, 2),
                        arange(out.shape[i_dim] - 2 * (n_i - 1), out.shape[i_dim], 2),
                    )
                ),
                i_dim,
            )
        return out


class SpectralMatern(SpectralModel):
    """
    Implement a spectral domain version of the Matern, which is approximate but much more efficient than the
    spatial domain version for non half integer values of the slope parameter.
    """

    rho = ModelParameter(default=1.0, bounds=(0, xp.inf), doc="range parameter")
    nu = ModelParameter(default=0.5, bounds=(0.5, xp.inf), doc="slope parameter")

    def __init__(self, rho=None, nu=None, name=None):
        super().__init__(rho, nu, name=name)

    def spectral_density(self, frequencies: xp.ndarray) -> xp.ndarray:
        """
        Implements the spectral density of the Matern.

        Parameters
        ----------
        frequencies
            shape (n1, ..., nk, d).

        Returns
        -------

        """
        ndim = frequencies.shape[-1]
        f2 = xp.sum(frequencies**2, -1)
        rho, nu = self.rho, self.nu
        term1 = (
            2**ndim
            * xp.pi ** (ndim / 2)
            * gamma(nu + ndim / 2)
            * (2 * nu) ** nu
            / (gamma(nu) * rho ** (2 * nu))
        )
        term2 = (2 * nu / rho**2 + 4 * xp.pi**2 * f2) ** (-nu - ndim / 2)
        return term1 * term2

    def _gradient(self, x: xp.ndarray):
        raise NotImplementedError()


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class NnSpectralModel(SpectralModel):
    """
    :class:`SpectralModel` whose spectral density is computed by a PyTorch
    neural network.

    The neural network is a ``torch.nn.Module`` that takes ``a + d`` inputs —
    ``a`` model parameter values followed by ``d`` frequency components — and
    outputs a single scalar (the spectral density value at that frequency).

    Subclasses define ``ModelParameter`` class attributes for the ``a`` model
    parameters, exactly as in any other covariance model.  The number of
    ``ModelParameter`` descriptors must match the number of parameter inputs
    expected by the neural network.

    This model requires the **torch** backend
    (``BackendManager.set_backend("torch")``) so that FFT operations in
    :meth:`call_on_rectangular_grid` preserve the autograd graph through the
    neural-network evaluation.

    Parameters
    ----------
    nn_module : torch.nn.Module
        Neural network mapping ``(a + d)``-dimensional input vectors to scalar
        spectral density values.
    *params
        Values for the ``ModelParameter`` descriptors, in declaration order.
    name : str, optional
        Model name.

    Examples
    --------
    >>> from debiased_spatial_whittle.models.base import ModelParameter
    >>> import torch
    >>>
    >>> class MyNnSpectralModel(NnSpectralModel):
    ...     rho = ModelParameter(default=1.0, bounds=(0, float("inf")))
    ...
    ...     def __init__(self, nn_module, rho=None, name=None):
    ...         super().__init__(nn_module, rho, name=name)
    >>>
    >>> net = torch.nn.Sequential(
    ...     torch.nn.Linear(1 + 2, 32), torch.nn.Tanh(),
    ...     torch.nn.Linear(32, 1), torch.nn.Softplus(),
    ... )
    >>> model = MyNnSpectralModel(net, rho=2.0)
    """

    def __init__(self, nn_module, *params, name: str = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("NnSpectralModel requires PyTorch to be installed.")
        self.nn_module = nn_module
        super().__init__(*params, name=name)

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Compute the covariance at the given lags.

        This overrides :meth:`SpectralModel.compute` to use
        :meth:`call_on_rectangular_grid` directly (building a grid from the lag
        shape and the model's deltas), avoiding the fragile grid-inference
        logic in the parent.  The covariance on the grid is obtained from the
        spectral density via the inverse-FFT path in
        :meth:`call_on_rectangular_grid`, and the values at the requested lags
        are extracted by indexing.

        Parameters
        ----------
        lags : xp.ndarray
            Lag vectors, shape ``(d, n1, ..., nk, 1)`` where ``d`` is the
            spatial dimension.  The trailing ``1`` is added by
            ``ModelInterface.__call__``.
        *params
            Model parameters (ignored — the NN uses ``self``'s parameters).

        Returns
        -------
        xp.ndarray
            Covariance values, shape ``(n1, ..., nk, 1)``.
        """
        from debiased_spatial_whittle.grids.base import RectangularGrid

        d = lags.shape[0]
        delta = tuple(1.0 for _ in range(d))

        # lags may have a trailing dim of 1 (added by __call__).
        if lags.shape[-1] == 1:
            lag_shape = lags.shape[1:-1]
        else:
            lag_shape = lags.shape[1:]

        # Build a grid matching the lag array's spatial shape.
        grid = RectangularGrid(lag_shape, delta=delta)

        # Covariance on the grid via spectral density -> inverse FFT.
        cov_on_grid = self.call_on_rectangular_grid(grid)

        # Map lags to indices (unit delta -> integer index).
        indices = []
        for dim in range(d):
            lags_dim = lags[dim]
            indices_dim = torch.round(torch.abs(lags_dim)).long()
            indices.append(indices_dim)

        result = cov_on_grid[tuple(indices)]
        return result

    def spectral_density(self, frequencies: xp.ndarray) -> xp.ndarray:
        """Evaluate the neural-network spectral density at the given frequencies.

        The frequency array has shape ``(n1, ..., nk, d)``.  It is flattened to
        ``(N, d)``, concatenated with broadcasted parameter values ``(N, a)``
        to form ``(N, a + d)`` inputs, and the NN is called once on the full
        batch.  The output is reshaped back to ``(n1, ..., nk)``.

        Parameters
        ----------
        frequencies : xp.ndarray
            Frequency grid, shape ``(n1, ..., nk, d)``.

        Returns
        -------
        xp.ndarray
            Spectral density values, shape ``(n1, ..., nk)``.
        """
        param_names = self._parameters
        a = len(param_names)

        # Detect the neural network's dtype and device from its parameters.
        try:
            nn_param = next(self.nn_module.parameters())
            nn_dtype = nn_param.dtype
            nn_device = nn_param.device
        except StopIteration:
            nn_dtype = torch.float64
            nn_device = None

        # Flatten frequencies to (N, d).
        freq_shape = frequencies.shape[:-1]  # (n1, ..., nk)
        d = frequencies.shape[-1]
        if not isinstance(frequencies, torch.Tensor):
            freqs_flat = torch.as_tensor(frequencies, dtype=nn_dtype, device=nn_device)
        else:
            freqs_flat = frequencies.to(dtype=nn_dtype, device=nn_device)
        freqs_flat = freqs_flat.reshape(-1, d)

        N = freqs_flat.shape[0]

        # Collect parameter values as torch tensors on the right device/dtype.
        param_values = []
        for pname in param_names:
            val = getattr(self, pname)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=torch.float64)
            val = val.to(dtype=nn_dtype, device=nn_device)
            param_values.append(val)

        # Assemble the full (N, a + d) input.
        if a > 0:
            param_rows = [v.reshape(1).expand(N) for v in param_values]
            params_flat = torch.stack(param_rows, dim=1)  # (N, a)
            nn_input = torch.cat([params_flat, freqs_flat], dim=1)
        else:
            nn_input = freqs_flat

        # Single batched forward pass (preserve autograd).
        sdf_flat = self.nn_module(nn_input)  # (N, 1) or (N,)
        if sdf_flat.dim() > 1 and sdf_flat.shape[-1] == 1:
            sdf_flat = sdf_flat.squeeze(-1)

        # Exponentiate to ensure a strictly positive spectral density.
        sdf_flat = torch.exp(sdf_flat)

        sdf = sdf_flat.reshape(*freq_shape)
        return sdf
