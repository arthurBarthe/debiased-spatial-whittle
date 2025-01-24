import sys

from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

import warnings
from debiased_spatial_whittle.periodogram import autocov
from typing import List

fftn, ifftn = BackendManager.get_fft_methods()
randn = BackendManager.get_randn()
arange = BackendManager.get_arange()


def prod_list(l: List[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])


# TODO make this work for 1-d and 3-d
def sim_circ_embedding(cov_func, shape):
    cov = autocov(cov_func, shape)
    f = prod_list(shape) * ifftn(cov)
    min_ = np.min(f)
    if min_ < 0:
        sys.exit(0)
        warnings.warn(f"Embedding is not positive definite, min value {min_}.")
    e = np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape)
    z = np.sqrt(np.maximum(f, 0)) * e
    z_inv = 1 / np.sqrt(prod_list(shape)) * np.real(fftn(z))
    for i, n in enumerate(shape):
        z_inv = np.take(z_inv, np.arange(n), i)
    return z_inv, min_


####NEW OOP VERSION
from typing import Tuple
from scipy.stats import multivariate_normal
from debiased_spatial_whittle.models import (
    CovarianceModel,
    TMultivariateModel,
    SquaredModel,
    ChiSquaredModel,
    BivariateUniformCorrelation,
    SeparableModel,
)
from debiased_spatial_whittle.grids import RectangularGrid


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
    >>> from debiased_spatial_whittle.models import ExponentialModel
    >>> from debiased_spatial_whittle.grids import RectangularGrid
    >>> model = ExponentialModel()
    >>> model.rho = 12.
    >>> model.sigma = 1.
    >>> grid = RectangularGrid((256, 128))
    >>> sampler = SamplerOnRectangularGrid(model, grid)
    >>> sample = sampler()
    >>> sample.shape
    (256, 128)
    """

    def __init__(self, model: CovarianceModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self.sampling_grid = grid
        self._f = None
        self._n_sims = 1
        self._i_sim = 0
        self._z = None
        try:
            self.f
        except:
            print("up-sampling")
            n = tuple(2 * n for n in self.grid.n)
            self.sampling_grid = RectangularGrid(n)  # may cause bugs?

    @property
    def n_sims(self):
        """number of simulations in each block computation."""
        return self._n_sims

    @n_sims.setter
    def n_sims(self, value: int):
        self._n_sims = value

    @property
    def f(self):
        """Spectral amplitudes of the covariance matrix on the circulant embedded grid."""
        if self._f is None:
            cov = self.sampling_grid.autocov(self.model)
            f = prod_list(self.sampling_grid.n) * ifftn(cov)
            f = np.real(f)
            min_ = np.min(f)
            if min_ <= -1e-5:
                raise ValueError(
                    f"Embedding is not positive definite, min value {min_}."
                )
            self._f = np.maximum(f, np.zeros_like(f))
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
            f = self.f
            shape = f.shape + (self.n_sims,)
            e = randn(*shape) + 1j * randn(*shape)
            f = np.expand_dims(f, -1)
            z = np.sqrt(np.maximum(f, np.zeros_like(f))) * e
            z_inv = (
                1
                / np.sqrt(np.array(prod_list(self.sampling_grid.n)))
                * np.real(fftn(z, axes=tuple(range(self.sampling_grid.ndim))))
            )
            for i, n in enumerate(self.grid.n):
                z_inv = np.take(z_inv, arange(n), i)
            self._z = z_inv * np.expand_dims(self.grid.mask, -1)
        result = self._z[..., self._i_sim % self._n_sims]
        self._i_sim += 1
        return result


class TSamplerOnRectangularGrid:
    """
    Class for the sampling of a t-multivariate random field
    """

    def __init__(self, model: TMultivariateModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self.gaussian_sampler = SamplerOnRectangularGrid(model.covariance_model, grid)

    def __call__(self):
        nu = self.model.nu_1.value
        chi = np.random.chisquare(nu) / nu
        # chi = np.random.chisquare(nu, self.grid.n) / nu  # this is a different model
        z = self.gaussian_sampler()
        return z / np.sqrt(chi)


class SquaredSamplerOnRectangularGrid:
    """
    Class for the sampling of a SquaredModel
    """

    def __init__(self, model: SquaredModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self.latent_sampler = SamplerOnRectangularGrid(self.model.latent_model, grid)

    def __call__(self):
        z = self.latent_sampler()
        return z**2


class ChiSquaredSamplerOnRectangularGrid:
    """
    Class for the sampling of a Chi Squared model on a rectangular grid
    """

    def __init__(self, model: ChiSquaredModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self.latent_sampler = SamplerOnRectangularGrid(self.model.latent_model, grid)

    def __call__(self):
        zs = []
        for i in range(self.model.dof_1.value):
            zs.append(self.latent_sampler())
        zs = np.stack(zs, axis=0)
        return np.sum(zs**2, axis=0)


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
        return np.prod(zs)

    def __call__(self):
        z = np.zeros(self.grid.n)
        for i in range(self.n_sim):
            z_i = self._unit_sample()
            z = i / (i + 1) * z + 1 / (i + 1) * z_i
        return z * np.sqrt(self.n_sim)


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
            f = np.real(f)
            min_ = np.min(f)
            if min_ <= -1e-5:
                sys.exit(0)
                warnings.warn(f"Embedding is not positive definite, min value {min_}.")
            self._f = np.maximum(f, np.zeros_like(f))
        return self._f

    def __call__(
        self, periodic: bool = False, return_spectral: bool = False
    ) -> np.ndarray:
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
        f = np.expand_dims(self.f, -1)
        z = np.sqrt(f) * e
        if return_spectral:
            return z
        z_inv = (
            1
            / np.sqrt(
                np.array(
                    [
                        self.grid.n_points,
                    ]
                )
            )
            * np.real(fftn(z, None, list(range(z.ndim - 1))))
        )
        if periodic:
            return z_inv
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        z_inv = np.reshape(z_inv, self.grid.n + (2,))
        return z_inv * self.grid.mask
