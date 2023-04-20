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
from .models import CovarianceModel, SeparableModel
from .grids import RectangularGrid


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
    def __call__(self, t_random_field:bool=False, df:int=10):
        f = self.f
        e = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape))
        z = np.sqrt(np.maximum(f, 0)) * e
        z_inv = 1 / np.sqrt(prod_list(self.grid.n)) * np.real(fftn(z))
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        z_inv = np.reshape(z_inv, self.grid.n)
        z = z_inv * self.grid.mask
        
        if t_random_field:
            if df == np.inf:
                chi = np.ones(self.grid.n_points)
            else:
                chi = np.random.chisquare(df, self.grid.n_points)/df
            
            z /= np.sqrt(chi.reshape(self.grid.n))
        return z


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
