from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.models.base import CovarianceModel
from debiased_spatial_whittle.grids.base import RectangularGrid

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

xp = BackendManager.get_backend()
fftn, ifftn = BackendManager.get_fft_methods()


class ConvolModel(CovarianceModel):
    """
    Covariance model defined by the autocorrelation of a sample.

    The covariance at a lag is obtained as the empirical autocorrelation of a
    sample (stored on the instance's grid), computed via FFT. This mirrors the
    spatial kernel computation (``spatial_kernel`` with ``m == (0, 0)``): the
    sample is zero-padded to ``(2*n1-1, ..., 2*nd-1)``, its squared modulus FFT
    is inverse-FFT'd and normalised, yielding the autocorrelation on the full
    lag grid with FFT layout (lag 0 at index 0, positive lags at 1, ..., n_i-1,
    negative lags at n_i, ..., 2*n_i-2).  The full array is zero-padded to the
    target grid's shape.

    Parameters
    ----------
    grid : RectangularGrid
        The grid on which the sample lives. Its deltas fix the lag spacing.
    sample : ndarray, optional
        Sample values, shape ``grid.n``. If ``None``, the sample must be
        provided by overriding ``_compute_sample``.
    name : str, optional
        Model name.

    Attributes
    ----------
    grid : RectangularGrid
        The instance's grid (deltas and shape define the autocorrelation grid).
    sample : ndarray or None
        The sample used to compute the autocorrelation.
    """

    def __init__(self, grid: RectangularGrid, sample=None, name: str = None):
        self.grid = grid
        self.sample = sample
        super().__init__(children=(), name=name)

    def _compute_sample(self):
        """Return the sample used to compute the autocorrelation.

        By default returns the ``sample`` attribute. Subclasses may override
        this to generate or transform the sample on the fly.
        """
        return self.sample

    def call_on_rectangular_grid(self, grid: RectangularGrid):
        """
        Compute the autocorrelation of the sample and embed it into ``grid``.

        The passed ``grid`` must have the same deltas as the instance's grid
        and be at least as large as ``2 * n_i - 1`` in every dimension.

        The autocorrelation is computed via FFT on the instance's grid shape
        ``(n1, ..., nd)``, producing an array of shape
        ``(2*n1-1, ..., 2*nd-1)`` with the standard FFT lag layout (lag 0 at
        index 0, positive lags at 1, ..., n_i-1, negative lags at n_i, ...,
        2*n_i-2).  The full array is zero-padded to the passed grid's shape
        ``(m1, ..., md)`` — no cutting is applied.

        Parameters
        ----------
        grid : RectangularGrid
            Target grid, with matching deltas and shape >= ``2*n_i - 1``.

        Returns
        -------
        ndarray
            Full autocorrelation, padded to ``grid.n``.
        """
        # Validate that deltas match
        if tuple(grid.delta) != tuple(self.grid.delta):
            raise ValueError(
                "Grid deltas must match the instance's grid deltas: "
                f"got {tuple(grid.delta)}, expected {tuple(self.grid.delta)}"
            )
        # Validate that the passed grid can hold the full autocorrelation
        n_inst = self.grid.n
        two_n = tuple(2 * s - 1 for s in n_inst)
        for i in range(self.grid.ndim):
            if grid.n[i] < two_n[i]:
                raise ValueError(
                    f"Grid dimension {i} has size {grid.n[i]}, "
                    f"smaller than the full autocorrelation size {two_n[i]}"
                )

        sample = self._compute_sample()
        if sample is None:
            raise ValueError("No sample available; set sample or override _compute_sample.")
        if not isinstance(sample, xp.ndarray):
            sample = xp.asarray(sample)

        n = self.grid.n

        # Autocorrelation via FFT, mirroring spatial_kernel m == (0, 0).
        normalization = xp.prod(xp.array(n))
        f = xp.abs(fftn(sample, two_n)) ** 2
        cg = ifftn(f)
        cg = xp.real(cg) / normalization

        # Pad with zeros to the passed grid's shape (m1, ..., md).
        pad_width = tuple((0, m - s) for m, s in zip(grid.n, two_n))
        result = xp.pad(cg, pad_width, mode="constant")
        return result

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Compute the covariance at the given lags.

        This delegates to ``call_on_rectangular_grid`` with a grid matching the
        lag shape and the instance's deltas, then extracts the values at the
        requested lag positions.  Since the full autocorrelation (with both
        positive and negative lags in FFT layout) is returned, signed lag
        indices are used directly: negative lags wrap to the high end of the
        array via ``index = lag + 2*n - 1``.

        Parameters
        ----------
        lags : ndarray
            Lag vectors, shape ``(d, n1, ..., nk, 1)`` where ``d`` is the
            spatial dimension. The trailing ``1`` is added by
            ``ModelInterface.__call__``.
        *params
            Model parameters (none for this model).

        Returns
        -------
        ndarray
            Covariance values, shape ``(n1, ..., nk, 1)``.
        """
        d = lags.shape[0]
        delta = self.grid.delta
        n_inst = self.grid.n
        two_n_inst = tuple(2 * s - 1 for s in n_inst)

        # Build a grid matching the lag array's spatial shape.
        # lags may have a trailing dim of 1 (added by __call__).
        if lags.shape[-1] == 1:
            lag_shape = lags.shape[1:-1]
        else:
            lag_shape = lags.shape[1:]
        grid_from_lags = RectangularGrid(lag_shape, delta=delta)

        # Full autocorrelation embedded in the lag grid.
        cov_on_grid = self.call_on_rectangular_grid(grid_from_lags)

        # Map lags to indices using FFT wrapping for negative lags.
        indices = []
        for dim in range(d):
            lags_dim = lags[dim]
            lag_idx = xp.round(lags_dim / delta[dim]).astype(int)
            # Negative lags wrap to the high end of the autocorrelation.
            lag_idx = xp.where(lag_idx < 0, lag_idx + two_n_inst[dim], lag_idx)
            indices.append(lag_idx)

        result = cov_on_grid[tuple(indices)]
        return result / torch.max(torch.abs(result))


class NnConvolModel(ConvolModel):
    """
    :class:`ConvolModel` whose sample is computed by a PyTorch neural network.

    The neural network is a ``torch.nn.Module`` that takes ``a + d`` inputs —
    ``a`` model parameter values followed by ``d`` integer grid coordinates —
    and outputs a single scalar (the sample value at that index).

    Subclasses define ``ModelParameter`` class attributes for the ``a`` model
    parameters, exactly as in any other covariance model.  The number of
    ``ModelParameter`` descriptors must match the number of parameter inputs
    expected by the neural network.

    This model requires the **torch** backend (``BackendManager.set_backend("torch")``)
    so that FFT operations in :meth:`call_on_rectangular_grid` preserve the
    autograd graph through the neural-network evaluation.

    Parameters
    ----------
    grid : RectangularGrid
        The grid on which the sample lives.
    nn_module : torch.nn.Module
        Neural network mapping ``(a + d)``-dimensional input vectors to scalar
        sample values.  Applied to a single index at a time; vectorised
        internally over all grid indices.
    *params
        Values for the ``ModelParameter`` descriptors, in declaration order.
    name : str, optional
        Model name.

    Examples
    --------
    >>> from debiased_spatial_whittle.models.base import ModelParameter
    >>> from debiased_spatial_whittle.grids.base import RectangularGrid
    >>> import torch
    >>>
    >>> class MyNnModel(NnConvolModel):
    ...     rho = ModelParameter(default=1.0, bounds=(0, float("inf")))
    ...
    ...     def __init__(self, grid, nn_module, rho=None, name=None):
    ...         super().__init__(grid, nn_module, rho, name=name)
    >>>
    >>> net = torch.nn.Sequential(
    ...     torch.nn.Linear(1 + grid.ndim, 32), torch.nn.Tanh(),
    ...     torch.nn.Linear(32, 1),
    ... )
    >>> model = MyNnModel(grid, net, rho=2.0)
    """

    def __init__(self, grid: RectangularGrid, nn_module, *params, name: str = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("NnConvolModel requires PyTorch to be installed.")
        self.grid = grid
        self.nn_module = nn_module
        self.sample = None
        # Bypass ConvolModel.__init__ (which passes no params) and call
        # CovarianceModel.__init__ directly so ModelParameter values are assigned.
        CovarianceModel.__init__(self, (), *params, name=name)

    def _compute_sample(self):
        """Evaluate the neural network at every grid index, vectorised.

        Builds an ``(N, a + d)`` input tensor — ``N`` being the total number
        of grid points — where each row is ``[param_1, ..., param_a, i_1, ...,
        i_d]``.  The neural network is called once on the full batch and the
        output is reshaped to the grid's spatial shape.
        """
        n = self.grid.n
        d = self.grid.ndim
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

        # Collect parameter values as torch tensors on the right device/dtype.
        param_values = []
        for pname in param_names:
            val = getattr(self, pname)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=torch.float64)
            val = val.to(dtype=nn_dtype, device=nn_device)
            param_values.append(val)

        # Build the coordinate grid of integer indices, rescaled to (0, 1),
        # flattened to (N, d).
        coords_1d = [
            torch.arange(n_i, dtype=nn_dtype, device=nn_device) / max(n_i - 1, 1) - 1 / 2
            for n_i in n
        ]
        mesh = torch.meshgrid(*coords_1d, indexing="ij")
        coords_flat = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # (N, d)

        N = coords_flat.shape[0]

        # Assemble the full (N, a + d) input.
        if a > 0:
            param_rows = [v.reshape(1).expand(N) for v in param_values]
            params_flat = torch.stack(param_rows, dim=1)  # (N, a)
            nn_input = torch.cat([params_flat, coords_flat], dim=1)
        else:
            nn_input = coords_flat

        # Single batched forward pass (no torch.no_grad — preserve autograd).
        sample_flat = self.nn_module(nn_input)  # (N, 1) or (N,)
        if sample_flat.dim() > 1 and sample_flat.shape[-1] == 1:
            sample_flat = sample_flat.squeeze(-1)

        # Exponentiate to obtain a non-negative sample, then subtract the mean
        # so the autocorrelation reflects spatial structure rather than the DC
        # level.  Without mean subtraction the NN can minimise the loss by
        # adjusting only the mean (a single scalar) without learning any spatial
        # variation, and the EP becomes insensitive to the parameters.
        sample = sample_flat.reshape(*n)
        sample = sample - xp.mean(sample)

        return sample
