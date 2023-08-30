import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd import elementwise_grad as egrad
from numpy import ndarray
from debiased_spatial_whittle.grids import RectangularGrid
import matplotlib.pyplot as plt
from debiased_spatial_whittle.periodogram import Periodogram, SeparableExpectedPeriodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, whittle, Estimator
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, Parameters
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, MCMC, GaussianPrior, Optimizer
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid
from debiased_spatial_whittle.bayes.funcs import transform
from numpy.linalg import inv


n = (64, 64)
grid = RectangularGrid(n)
N = grid.n_points

lags = grid.lags_unique

model = ExponentialModel()
params = np.array([8., 1.])
model.rho = params[0]
model.sigma = params[1]
model.nugget=0.1

dw = DeWhittle(np.ones(n), grid, ExponentialModel(), nugget=0.1, transform_func=transform)

def exp_model(params: ndarray, lags):
    rho, sigma = np.exp(params)
    nugget = 0.1
    
    d = np.sqrt(sum((lag**2 for lag in lags)))
    nugget_effect = nugget * np.all(lags == 0, axis=0)
    
    acf = sigma**2 * np.exp(- d / rho) + nugget_effect
    return acf


grads = model.gradient(grid.lags_unique, Parameters([model.rho, model.sigma ]))
d_rho, d_sigma = grads['rho'], grads['sigma']

jac = jacobian(exp_model)(np.log(params), lags).T   # Theoretical deriv. with log transform
d_rho2, d_sigma_2 = jac

print(np.allclose(d_rho, d_rho2))
print(np.allclose(d_sigma, d_sigma_2))

jac_params = np.diag(jacobian(transform)(np.log(params), inv=True))

print(np.allclose(np.stack(grads.values()) *  jac_params[:,None,None], jac))
stop
# print(np.allclose(d_rho, d_rho2 *   ))
print(np.allclose(d_sigma, d_sigma_2 * grad(transform)(np.log(params[1]))))


stop

cov3 = exp_model(params, *lags.reshape(2, -1)).reshape(lags.shape[1:])
print(np.allclose(cov,cov3))
d_cov = egrad(exp_model2)(params, *lags.reshape(2, -1))


stop



def exp_model(params: ndarray, lags):
    rho, sigma = params
    nugget = 0.1
    
    d = np.sqrt(sum((lag**2 for lag in lags)))
    nugget_effect = nugget * np.all(lags == 0, axis=0)
    
    acf = sigma**2 * np.exp(- d / rho) + nugget_effect
    return acf

def exp_model2(params: ndarray, lags):
    rho, sigma = np.exp(params)
    nugget = 0.1
    
    d = np.sqrt(sum((lag**2 for lag in lags)))
    nugget_effect = nugget * np.all(lags == 0, axis=0)
    
    acf = sigma**2 * np.exp(- d / rho) + nugget_effect
    return acf


jac = jacobian(exp_model)(params, lags).T
d_rho, d_sigma = jac

jac = jacobian(exp_model2)(np.log(params), lags).T
d_rho2, d_sigma_2 = jac

print(np.allclose(d_rho, d_rho2))
print(np.allclose(d_sigma, d_sigma_2))

print(np.allclose(d_rho, d_rho2 * grad(transform)(params[0], inv=False)  ))
print(np.allclose(d_sigma, d_sigma_2 * grad(transform)(np.log(params[1]))))



