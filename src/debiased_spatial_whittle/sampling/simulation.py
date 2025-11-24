import sys, warnings
from typing import Tuple
from scipy.stats import multivariate_normal
from debiased_spatial_whittle.models.base import (
    CovarianceModel,
    TMultivariateModel,
    SquaredModel,
    ChiSquaredModel,
    SeparableModel,
)
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.backend import BackendManager


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
        self, model: CovarianceModel, grid: RectangularGrid, exact: bool = True
    ):
        self.model = model
        self.grid = grid
        self.sampling_grid = grid
        self._f = None
        self._n_sims = 1
        self._i_sim = 0
        self._z = None
        self.exact = exact
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
            if not self.exact:
                cov *= self.sampling_grid.spatial_kernel()
            f = prod_list(self.sampling_grid.n) * ifftn(cov)
            f = xp.real(f)
            min_, max_ = xp.min(f), xp.max(f)
            if min_ <= -1e-2:
                print(min_, max_)
                raise ValueError(
                    f"Embedding is not positive definite, min value {min_}."
                )
            self._f = xp.maximum(f, xp.zeros_like(f))
        return self._f

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
        return result


from numpy.linalg import eigh


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

    def compute_spectral_decomposition(self):
        # cov shape (2 * n1 - 1, 2 * n2 - 1, p, p)
        cov = self.sampling_grid.autocov(self.model)
        f = prod_list(self.sampling_grid.n) * ifftn(cov, axes=self.spatial_axes)
        return eigh(f)

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
        return self._sample()


class SamplerSeparable:
    """
    Class for approximate sampling of Separable models.
    """

    def __init__(self, model: SeparableModel, grid: RectangularGrid, n_sim: int = 100):
        assert isinstance(model, SeparableModel)
        self.model = model
        self.grid = grid
        self.n_sim = n_sim
        self.samplers = self._setup_samplers()

    def _setup_samplers(self):
        samplers = []
        for model, dims in zip(self.model.models, self.model.dims):
            sampler = SamplerOnRectangularGrid(model, self.grid.separate(dims))
            samplers.append(sampler)
        return samplers

    def _unit_sample(self):
        zs = []
        for sampler in self.samplers:
            zs.append(sampler())
        return xp.prod(zs)

    def __call__(self):
        z = xp.zeros(self.grid.n)
        for i in range(self.n_sim):
            z_i = self._unit_sample()
            z = i / (i + 1) * z + 1 / (i + 1) * z_i
        return z * xp.sqrt(self.n_sim)


class SamplerBUCOnRectangularGrid:
    """
    Class to sample from the BivariateUniformCorrelation model on a rectangular grid with nvars=2.

    Attributes
    ----------
    grid: RectangularGrid
        Sampling grid. Should have attribute nvars=2.

    model: BivariateUniformCorrelation
        Bivariate covariance model

    f: ndarray
        Spectral amplitudes
    """

    def __init__(self, model: BivariateUniformCorrelation, grid: RectangularGrid):
        assert isinstance(model, BivariateUniformCorrelation)
        self.model = model
        self.grid = grid
        self.e_dist = multivariate_normal([0, 0], [[1, model.r], [model.r, 1]])
        self._f = None

    @property
    def f(self):
        if self._f is None:
            cov = self.grid.autocov(self.model.base_model)
            f = prod_list(self.grid.n) * ifftn(cov)
            f = xp.real(f)
            min_ = xp.min(f)
            if min_ <= -1e-5:
                sys.exit(0)
                warnings.warn(f"Embedding is not positive definite, min value {min_}.")
            self._f = xp.maximum(f, xp.zeros_like(f))
        return self._f

    def __call__(
        self, periodic: bool = False, return_spectral: bool = False
    ) -> xp.ndarray:
        """
        Sample a realization.

        Parameters
        ----------
        periodic
            if true, returns a periodic sample on an embedding grid

        return_spectral
            if true, returns the spectral amplitudes as well

        Returns
        -------
        sample: ndarray
            shape (n1, ..., nd, 2) where the last dimension indexes the two variates.
        """
        f = self.f
        e = self.e_dist.rvs(size=f.shape + (2,))
        e = BackendManager.convert(e)
        e[..., -1] *= self.model.f
        e = e[..., 0, :] + 1j * e[..., 1, :]
        f = xp.expand_dims(self.f, -1)
        z = xp.sqrt(f) * e
        if return_spectral:
            return z
        z_inv = (
                1
                / xp.sqrt(
                xp.array(
                    [
                        self.grid.n_points,
                    ]
                )
            )
                * xp.real(fftn(z, None, list(range(z.ndim - 1))))
        )
        if periodic:
            return z_inv
        for i, n in enumerate(self.grid.n):
            z_inv = xp.take(z_inv, xp.arange(n), i)
        z_inv = xp.reshape(z_inv, self.grid.n + (2,))
        return z_inv * self.grid.mask
