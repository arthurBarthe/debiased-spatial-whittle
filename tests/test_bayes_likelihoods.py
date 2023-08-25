import autograd.numpy as np
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel, SquaredModel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, MCMC, GaussianPrior, Optimizer
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid

from debiased_spatial_whittle.bayes.funcs import transform
from numpy.testing import assert_allclose

from autograd import grad, hessian

np.random.seed(5325234)
inv = np.linalg.inv

def test_likelihood_dewhittle_grad():

    n = (64,64)
    params = np.array([8.,1.])
    model = SquaredExponentialModel()
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    
    x = np.log(params)
    ll_grad = grad(dw)(x)
    ll_hess = hessian(dw)(x)
    assert np.all(np.isfinite(ll_grad))
    assert np.all(np.isfinite(ll_hess))


def test_likelihood_whittle_grad():

    n = (64,64)
    params = np.array([8.,1.])
    model = SquaredExponentialModel()
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    whittle = Whittle(z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)  
    
    x = np.log(params)
    ll_grad = grad(whittle)(x)
    ll_hess = hessian(whittle)(x)
    assert np.all(np.isfinite(ll_grad))
    assert np.all(np.isfinite(ll_hess))

def test_likelihood_gauss_grad():

    n = (16,16)
    params = np.array([2.,1.])
    model = SquaredExponentialModel()
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    gauss = Gaussian(z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)  
    
    x = np.log(params)
    ll_grad = grad(gauss)(x)
    ll_hess = hessian(gauss)(x)  # TODO: this is slow!
    assert np.all(np.isfinite(ll_grad))
    assert np.all(np.isfinite(ll_hess))
    
if __name__ == '__main__':
    test_likelihood_dewhittle_grad()
    test_likelihood_whittle_grad()
    test_likelihood_gauss_grad()
    
    