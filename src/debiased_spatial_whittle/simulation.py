import sys

from .backend import BackendManager
np = BackendManager.get_backend()

import warnings
from .periodogram import autocov
from typing import List

fftn = np.fft.fftn
ifftn = np.fft.ifftn


def prod_list(l: List[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])

#TODO make this work for 1-d and 3-d
def sim_circ_embedding(cov_func, shape):
    cov = autocov(cov_func, shape)
    f = prod_list(shape) * ifftn(cov)
    min_ = np.min(f)
    if min_ < 0:
        sys.exit(0)
        warnings.warn(f'Embedding is not positive definite, min value {min_}.')
    e = np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape)
    z = np.sqrt(np.maximum(f, 0)) * e
    z_inv = 1 / np.sqrt(prod_list(shape)) * np.real(fftn(z))
    for i, n in enumerate(shape):
        z_inv = np.take(z_inv, np.arange(n), i)
    return z_inv, min_



####NEW OOP VERSION
from typing import Tuple
from debiased_spatial_whittle.models import CovarianceModel, SeparableModel, BivariateUniformCorrelation
from debiased_spatial_whittle.grids import RectangularGrid


def prod_list(l: Tuple[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])


class SamplerOnRectangularGrid:
    """Class that allows to define samplers for Rectangular grids, for which
    fast exact sampling can be achieved via circulant embeddings and the use of the
    Fast Fourier Transform."""

    def __init__(self, model: CovarianceModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self._f = None

    @property
    def f(self):
        if self._f is None:
            cov = self.grid.autocov(self.model)
            f = prod_list(self.grid.n) * ifftn(cov)
            min_ = np.min(f)
            if min_ <= -1e-5:
                sys.exit(0)
                warnings.warn(f'Embedding is not positive definite, min value {min_}.')
            self._f = np.maximum(f, 0)
        return self._f

    # TODO make this work for 1-d and 3-d
    def __call__(self, periodic: bool = False):
        f = self.f
        e = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape))
        z = np.sqrt(np.maximum(f, 0)) * e
        z_inv = 1 / np.sqrt(prod_list(self.grid.n)) * np.real(fftn(z))
        if periodic:
            return z_inv
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        z_inv = np.reshape(z_inv, self.grid.n)
        return z_inv * self.grid.mask


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

from scipy.stats import multivariate_normal

class SamplerCorrelatedOnRectangularGrid:
    """Class that allows to define samplers for Rectangular grids, for which
    fast exact sampling can be achieved via circulant embeddings and the use of the
    Fast Fourier Transform."""

    def __init__(self, model: CovarianceModel, grid: RectangularGrid, correlation: float):
        self.model = model
        self.grid = grid
        self.e_dist = multivariate_normal(np.zeros(2), [[1, correlation], [correlation, 1]])
        self._f = None

    @property
    def f(self):
        if self._f is None:
            cov = self.grid.autocov(self.model)
            f = prod_list(self.grid.n) * ifftn(cov)
            min_ = np.min(f)
            if min_ <= -1e-5:
                sys.exit(0)
                warnings.warn(f'Embedding is not positive definite, min value {min_}.')
            self._f = np.maximum(f, 0)
        return self._f

    # TODO make this work for 1-d and 3-d
    def __call__(self, periodic: bool = False):
        f = np.expand_dims(self.f, -1)
        e = self.e_dist.rvs(size=f.shape + (2, ))
        e = e[..., 0, :] + 1j * e[..., 1, :]
        z = np.sqrt(np.maximum(f, 0)) * e
        z_inv = 1 / np.sqrt(self.grid.n_points) * np.real(fftn(z, axes=list(range(z.ndim - 1))))
        if periodic:
            return z_inv
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        z_inv = np.reshape(z_inv, self.grid.n + (2, ))
        return z_inv * np.expand_dims(self.grid.mask, -1)


class SamplerBUCOnRectangularGrid:
    """
    Class to sample from the BivariateUniformCorrelation model on a rectangular grid.
    """
    def __init__(self, model: BivariateUniformCorrelation, grid: RectangularGrid):
        assert isinstance(model, BivariateUniformCorrelation)
        self.model = model
        self.grid = grid
        self.e_dist = multivariate_normal(np.zeros(2), [[1, model.r_0.value], [model.r_0.value, 1]])
        self._f = None

    @property
    def f(self):
        if self._f is None:
            cov = self.grid.autocov(self.model.base_model)
            f = prod_list(self.grid.n) * ifftn(cov)
            min_ = np.min(f)
            if min_ <= -1e-5:
                sys.exit(0)
                warnings.warn(f'Embedding is not positive definite, min value {min_}.')
            self._f = np.maximum(f, 0)
        return self._f

    # TODO make this work for 1-d and 3-d
    def __call__(self, periodic: bool = False, return_spectral: bool = False):
        f = np.expand_dims(self.f, -1)
        e = self.e_dist.rvs(size=f.shape + (2,))
        e[..., -1] *= self.model.f_0.value
        e = e[..., 0, :] + 1j * e[..., 1, :]
        z = np.sqrt(f) * e
        if return_spectral:
            return z
        z_inv = 1 / np.sqrt(self.grid.n_points) * np.real(fftn(z, axes=list(range(z.ndim - 1))))
        if periodic:
            return z_inv
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        z_inv = np.reshape(z_inv, self.grid.n + (2,))
        return z_inv * np.expand_dims(self.grid.mask, -1)









def test_simulation_1d():
    from numpy.random import seed
    from .grids import RectangularGrid
    from .models import ExponentialModel
    import matplotlib.pyplot as plt
    seed(1712)
    model = ExponentialModel()
    model.rho = 2
    model.sigma = 1
    grid1 = RectangularGrid((16, 1))
    grid2 = RectangularGrid((16, ))
    sampler1 = SamplerOnRectangularGrid(model, grid1)
    sampler2 = SamplerOnRectangularGrid(model, grid2)
    z1 = sampler1()
    seed(1712)
    z2 = sampler2()
    plt.figure()
    plt.plot(z1)
    plt.plot(z2, '*')
    plt.show()
    assert True
