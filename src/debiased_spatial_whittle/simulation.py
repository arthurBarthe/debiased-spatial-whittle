import sys
from typing import Union
import numpy as np
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
from debiased_spatial_whittle.models import CovarianceModel, SeparableModel, TMultivariateModel, SquaredModel, \
    PolynomialModel, ChiSquaredModel
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
        self.sampling_grid = grid
        self._f = None
        try:
            self.f

        except:
            print('up-sampling')
            n = tuple(2*n for n in self.grid.n)
            self.sampling_grid = RectangularGrid(n)    # may cause bugs?

    @property
    def f(self):
        if self._f is None:
            cov = self.sampling_grid.autocov(self.model)
            f = prod_list(self.sampling_grid.n) * ifftn(cov)
            min_ = np.min(f)
            if min_ <= -1e-5:
                raise ValueError(f'Embedding is not positive definite, min value {min_}.')
                
            self._f = np.maximum(f, 0)
        return self._f

    # TODO make this work for 1-d and 3-d
    def __call__(self):
        f = self.f
        e = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape))
        z = np.sqrt(np.maximum(f, 0)) * e
        z_inv = 1 / np.sqrt(prod_list(self.sampling_grid.n)) * np.real(fftn(z))
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        # print(z_inv.shape)
        z_inv = np.reshape(z_inv, self.grid.n)
        z = z_inv * self.grid.mask
        return z


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
        return z ** 2


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


class PolynomialSamplerOnRectangularGrid:
    """
    Class for the sampling of a SquaredModel
    """
    def __init__(self, model: PolynomialModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self.latent_sampler = SamplerOnRectangularGrid(self.model.latent_model, grid)

    def __call__(self):
        z = self.latent_sampler()
        a = self.model.a_1.value
        b = self.model.b_1.value
        return a * z**2 + b * z
    


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


def test_upsampling():
    from numpy.random import seed
    import matplotlib.pyplot as plt
    from debiased_spatial_whittle.grids import RectangularGrid
    from debiased_spatial_whittle.models import ExponentialModel
    from debiased_spatial_whittle.plotting_funcs import plot_marginals
    from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian

    model = ExponentialModel()
    model.rho = 40
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid((128,128))
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    plt.figure()
    plt.imshow(z, cmap='Spectral')
    plt.show()
    
    params = np.log([40,1])
    
    dw = DeWhittle(z, grid, ExponentialModel(), nugget=0.1)
    dw.fit(None, prior=False)
    # stop
    MLEs = dw.sim_MLEs(params, niter=500)
    plot_marginals([MLEs], params)
    
def t_rf_test():
    from debiased_spatial_whittle.models import ExponentialModel
    from debiased_spatial_whittle.bayes import DeWhittle
    from debiased_spatial_whittle.plotting_funcs import plot_marginals
    np.random.seed(18979125)
    n=(64,64)
    grid = RectangularGrid(n)
    t_model = TMultivariateModel(ExponentialModel())
    t_model.nugget_0 = 0.1
    t_model.nu_1 = 5.
    print(t_model)
    params = np.log([10.,1.])
    dw = DeWhittle(np.ones(n), grid, t_model, nugget=0.)   # TODO: wrong nugget name
    MLEs_t = dw.sim_MLEs(np.exp(params), niter=1000)

    model = ExponentialModel()
    model.nugget = 0.1
    dw_gauss = DeWhittle(np.ones(n), grid, model, nugget=0.)   # TODO: wrong nugget name
    MLEs_g = dw_gauss.sim_MLEs(np.exp(params), niter=1000)
    
    plot_marginals([MLEs_t, MLEs_g], params, density_labels=['t', 'gauss'])
   
        