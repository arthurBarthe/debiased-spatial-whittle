from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

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

def test_likelihood_update_model_param():

    n = (64,64)
    model = SquaredExponentialModel()
    model.rho = 8
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    dw = DeWhittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    
    new_params = np.array([10.,2.])
    dw.update_model_params(new_params)
    assert np.allclose(dw.model.param_values[:-1], new_params)
    
def test_likelihood_transform():

    n = (64,64)
    model = SquaredExponentialModel()

    params = np.array([10.,2.])
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    dw = DeWhittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    whittle = Whittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    
    x = params.copy()
    assert_allclose(dw.transform(x, inv=True), np.exp(x))
    assert_allclose(dw.transform(x, inv=False), np.log(x))
    
    assert_allclose(whittle.transform(x, inv=True), np.exp(x))
    assert_allclose(whittle.transform(x, inv=False), np.log(x))
    
def test_likelihood_no_transform():

    n = (64,64)
    model = SquaredExponentialModel()

    params = np.array([10.,2.])
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    dw = DeWhittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=None)
    whittle = Whittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=None)
    
    x = params.copy()
    assert_allclose(dw.transform(x, inv=True), x)
    assert_allclose(dw.transform(x, inv=False), x)
    
    assert_allclose(whittle.transform(x, inv=True), x)
    assert_allclose(whittle.transform(x, inv=False), x)
    
def test_likelihood_cov_func():

    n = (64,64)
    model = SquaredExponentialModel()

    params = np.array([10.,2.])
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    dw = DeWhittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    
    cov_true = model(grid.lags_unique)
    cov_test = dw.cov_func(params)
    assert_allclose(cov_true, cov_test)
    
    
def test_likelihood_sim_z():

    n = (64,64)
    model = SquaredExponentialModel()
    model.rho = 8
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    params = np.array([8.,1.])
    dw = DeWhittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    z = dw.sim_z(params)
    
    assert np.all(np.isfinite(z))
    
    
def test_likelihood_sim_MLEs():

    n = (64,64)
    model = SquaredExponentialModel()
    model.rho = 8
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid(n)
    
    params = np.array([8.,1.])
    dw = DeWhittle(np.ones(n), grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform)
    MLEs = dw.sim_MLEs(params, niter=10, print_res=False)
    
    assert np.all(np.isfinite(MLEs))


if __name__ == '__main__':
    test_likelihood_update_model_param()
    test_likelihood_transform()
    test_likelihood_no_transform()
    test_likelihood_cov_func()
    test_likelihood_sim_z()
    test_likelihood_sim_MLEs()


