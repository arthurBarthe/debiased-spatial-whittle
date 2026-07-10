import sys, warnings
from typing import Tuple

from charset_normalizer.md import lru_cache
from scipy.stats import multivariate_normal
from debiased_spatial_whittle.models.base import CovarianceModel, SeparableModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.sampling.samples import SampleOnRectangularGrid

xp = BackendManager.get_backend()
fftn, ifftn = BackendManager.get_fft_methods()
randn = BackendManager.get_randn()
arange = BackendManager.get_arange()


def prod_list(l: Tuple[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])


class SamplerOnRectangularGrid:
    """
    Class that allows to define efficient samplers on rectangular grids for fixed models.

    Attributes
    ----------
    model: CovarianceModel
        Covariance model used for sampling.

    grid: RectangularGrid
        Grid on which we wish to sample

    n_sims: int
        Simulations can be carried out in 'blocks'. This parameter allows to choose how many
        i.i.d. samples are generated in each block computation.

    f: ndarray
        Spectral amplitudes

    Notes
    -----
    This sampler accounts for the grid's mask by setting missing values to zero.

    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
    >>> from debiased_spatial_whittle.grids.base import RectangularGrid
    >>> model = ExponentialModel(rho=12., sigma=1.)
    >>> grid = RectangularGrid((256, 128))
    >>> sampler = SamplerOnRectangularGrid(model, grid)
    >>> sample = sampler()
    >>> sample.shape
    (256, 128)
    """

    def __init__(
        self, model: CovarianceModel, grid: RectangularGrid, tol: float = 0.01
    ):
        """
        Parameters
        ----------
        model: CovarianceModel
            Model from which we wish to sample

        grid: RectangularGrid
            Grid on which we wish to sample
        tol: float
            Tolerance level. The circulant embedding method embeds the covariance matrix into a circulant matrix, which
            is then diagonal in the Fourier domain. However, the circulant embedding might not be non-negative definite.
            This results in negative values on the diagonal. We compute the absolute value of the sum of negative values,
            and the sum of positive values. If the ratio of the two is greater than the tolerance level, we raise
            an error.
        """
        self.model = model
        self.grid = grid
        self.sampling_grid = grid
        self._f = None
        self._n_sims = 1
        self._i_sim = 0
        self._z = None
        self.tol = tol
        try:
            self.spectral_amplitudes
        except:
            print("up-sampling")
            n = tuple(2 * n for n in self.grid.n)
            self.sampling_grid = RectangularGrid(n, grid.delta)

    @property
    def model(self) -> CovarianceModel:
        """Model from which we sample"""
        return self._model

    @model.setter
    def model(self, value: CovarianceModel):
        self._model = value
        self._f = None
        self._i_sim = 0

    @property
    def grid(self) -> RectangularGrid:
        """Sampling grid"""
        return self._grid

    @grid.setter
    def grid(self, value: RectangularGrid):
        self._grid = value
        self._f = None
        self._i_sim = 0

    @property
    def n_sims(self):
        """number of simulations in each block computation. By increasing this value, one allows parallel simulation,
        at the expense of increased memory usage."""
        return self._n_sims

    @n_sims.setter
    def n_sims(self, value: int):
        self._n_sims = value

    @property
    def spectral_amplitudes(self):
        """Spectral amplitudes of the covariance matrix on the circulant embedded grid."""
        if self._f is None:
            cov = self.sampling_grid.autocov(self.model)
            f = prod_list(self.sampling_grid.n) * ifftn(cov)
            f = xp.real(f)
            if self._get_level(f) > self.tol:
                raise ValueError(
                    f"Embedding is not positive definite, {self._get_level(f)} > {self.tol}"
                )
            self._f = xp.maximum(f, xp.zeros_like(f))
        return self._f

    def _get_level(self, amplitudes: xp.ndarray):
        negative = xp.sum(xp.abs(amplitudes[amplitudes < 0]))
        positive = xp.sum(amplitudes[amplitudes > 0])
        return negative / positive

    def __call__(self):
        """
        Samples a realization of a Gaussian Process specified by
        the provided covariance model, on the provided rectangular grid.

        Returns
        -------
        sample: ndarray
            Sample values corresponding to the grid and covariance model. Shape is equal to the n attribute of grid.

        Raises
        ------
        ValueError
            If a non-negative definite circulant embedding could not be achieved. In that case a solution
            is to increase the grid size.
        """
        if self._i_sim % self.n_sims == 0:
            f = self.spectral_amplitudes
            shape = f.shape + (self.n_sims,)
            e = randn(*shape) + 1j * randn(*shape)
            f = xp.expand_dims(f, -1)
            z = xp.sqrt(xp.maximum(f, xp.zeros_like(f))) * e
            z_inv = (
                    1
                    / xp.sqrt(xp.array(prod_list(self.sampling_grid.n)))
                    * xp.real(fftn(z, axes=tuple(range(self.sampling_grid.ndim))))
            )
            for i, n in enumerate(self.grid.n):
                z_inv = xp.take(z_inv, arange(n), i)
            self._z = z_inv * xp.expand_dims(self.grid.mask, -1)
        result = self._z[..., self._i_sim % self._n_sims]
        self._i_sim += 1
        return SampleOnRectangularGrid(self.grid, result)


class MultivariateSamplerOnRectangularGrid:
    """
    Implements circulant embedding for multivariate random fields, as proposed by Chan & Wood (1999).
    """

    def __init__(self, model: CovarianceModel, grid: RectangularGrid, p: int):
        """
        Parameters
        ----------
        model
            Model from which we sample
        grid
            Sampling grid
        p
            Number of variates of the model
        """
        self.model = model
        self.grid = grid
        self.p = p
        self.sampling_grid = grid

    @property
    def model(self) -> CovarianceModel:
        """Model from which we sample"""
        return self._model

    @model.setter
    def model(self, value: CovarianceModel):
        self._model = value

    @property
    def grid(self) -> RectangularGrid:
        """Sampling grid"""
        return self._grid

    @grid.setter
    def grid(self, value: RectangularGrid):
        self._grid = value

    @property
    def spatial_axes(self):
        return tuple(range(self.grid.ndim))

    @lru_cache
    def compute_spectral_decomposition(self):
        # cov shape (2 * n1 - 1, 2 * n2 - 1, p, p)
        cov = self.sampling_grid.autocov(self.model)
        f = prod_list(self.sampling_grid.n) * ifftn(cov, axes=self.spatial_axes)
        return xp.linalg.eigh(f)

    def _sample(self):
        # lambdas shape (p, ), r_matrix shape (p, p)
        lambdas, r_matrix = self.compute_spectral_decomposition()
        shape = self.sampling_grid.n + (1, self.p)
        # e shape (n1, ..., nd, p, 1)
        e = randn(*lambdas.shape) + 1j * randn(*lambdas.shape)
        y = xp.expand_dims(xp.sqrt(lambdas) * e, -1)
        y = xp.matmul(r_matrix, y)
        w = (
                1
                / xp.sqrt(xp.array(prod_list(self.sampling_grid.n)))
                * xp.real(fftn(y, axes=self.spatial_axes))
        )
        for i, n in enumerate(self.grid.n):
            w = xp.take(w, arange(n), i)
        # remove extra dimensions
        # TODO dirty, clean this somehow
        w = xp.squeeze(w, -1)
        return w * self.grid.mask

    def __call__(self) -> xp.ndarray:
        """
        Generate a realization from the specified covariance model on the grid.

        Returns
        -------
        sample
            Simulated sample.
        """
        sample = self._sample()
        return SampleOnRectangularGrid(self.grid, sample)
