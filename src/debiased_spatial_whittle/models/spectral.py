from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.models.base import CovarianceModel, ModelParameter
from abc import ABCMeta, abstractmethod

xp = BackendManager.get_backend()
fftn, ifftn = BackendManager.get_fft_methods()
arange = BackendManager.get_arange()
gamma = BackendManager.get_gamma()


class SpectralModel(CovarianceModel):
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

    def __call__(self, lags: xp.ndarray):
        """
        Compute an approximation to the covariance function evaluated at the passed lags based on the spectral
        density function.

        Currently, not implemented: we only provide an implementation in the case where the lags are those
        of a grid, cf method call_on_rectangular_grid.

        Parameters
        ----------
        lags
            shape (d, n1, n2, ..., nk)

        Returns
        -------
        cov
            shape (n1, ..., nk). Approximate values of the covariance function
        """
        raise NotImplementedError()

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

    def __init__(self, *args, **kwargs):
        super(SpectralMatern, self).__init__(*args, **kwargs)

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
