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
