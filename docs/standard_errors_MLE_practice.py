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

from numpy.linalg import inv


n = (64, 64)
grid = RectangularGrid(n)
N = grid.n_points

lags = grid.lags_unique

model = ExponentialModel()
params =  np.array([8., 1.]) + np.random.rand(2)
model.rho = params[0]
model.sigma = params[1]
model.nugget=0.1

dw = DeWhittle(np.ones(n), grid, ExponentialModel(), nugget=0.1, transform_func=None)


p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
H = d.fisher(model, Parameters([model.rho, model.sigma ]))
print(H)
assert np.all(np.diag(H) >= 0)

H2 = dw.fisher(params)


# TODO: test with log parameterization vs simulation!!!
print(np.allclose(H, H2 * (2/dw.n_points) ))

# J = d.jmatrix(model, Parameters([model.rho, model.sigma ]))
# approx_sandwich = inv(H) @ J @ inv(H)
# approx_sandwich = d.variance_of_estimates(model, Parameters([model.rho, model.sigma ]) )


stop

M = 5000
p = len(params)
grads = np.zeros((M, p))
hessians = np.zeros((M, p, p))
for i in range(M):
    print(f'\ri={i+1}', end='')
    z = dw.sim_z(params)
    
    # plt.imshow(z)
    # plt.show()
    grads[i] = grad(dw)(params, z=z, constant='whittle')          # TODO: CONSTANTS ON DEWHITTLE!!!!
    hessians[i] = hessian(dw)(params, z=z, constant='whittle')

Jhat = np.cov((2/N) * grads.T)
print()
print( Jhat )
print( -2 * np.mean(hessians,axis=0)/ grid.n_points )

sandwich = inv(H) @ Jhat @ inv(H)   # TODO: H is negative?
print(inv(sandwich))
print(sandwich)

niter=500
MLEs = dw.sim_MLEs(params, niter=niter, print_res=False)
print(np.cov(MLEs.T))





stop












cov2 = exp_model(params, lags.reshape(2, -1)).reshape(lags.shape[1:])
print(np.allclose(cov,cov2))


def exp_model2(params: ndarray, lag1, lag2):
    rho, sigma = np.exp(params)
    nugget = 0.1
    
    d = np.sqrt(sum((lag**2 for lag in [lag1,lag2])))
    nugget_effect = nugget * ((lag1 == 0) & (lag2 == 0))
    
    acf = sigma**2 * np.exp(- d / rho) + nugget_effect
    return acf

cov3 = exp_model2(params, *lags.reshape(2, -1)).reshape(lags.shape[1:])
print(np.allclose(cov,cov3))
d_cov = egrad(exp_model2)(params, *lags.reshape(2, -1))


def f(param1, param2, xs):
    return (param1 + param2) * xs




def d_cov_func(covfunc, params, argnum=0):
    gradcov = grad(covfunc, argnum)
    def d_cov(lags):
        lags = np.array(lags)
        d,*shape = lags.shape 
        xs       = lags.reshape(d,-1).T
        d_cov_xs = np.array([gradcov(params,x) for x in xs])
        return np.squeeze(d_cov_xs.reshape(*shape,-1).T)
    return d_cov

d_covfunc = d_cov_func(exp_model, params)

d_rho, d_sigma = d_covfunc(lags)
plt.imshow(d_rho)
plt.show()
plt.imshow(d_sigma)
plt.show()

jac = jacobian(exp_model)(params, lags).T

d_rho2, d_sigma2 = jac

plt.imshow(d_rho)
plt.show()
plt.imshow(d_sigma)
plt.show()

print(np.allclose(d_rho,d_rho2))
print(np.allclose(d_sigma,d_sigma2))

# def sphere_potential(xy):
#     R = 0.5
#     r = np.sqrt(np.sum(xy**2, axis=-1))
#     return np.minimum(1/r, 1/R)

# def potential(xy):
#     sphere_pos = np.array([0., 1.])
#     return (  sphere_potential(xy - sphere_pos)
#             - sphere_potential(xy + sphere_pos))

# X, Y = np.meshgrid(*[np.linspace(-2, 2, 50)]*2)
# xy_grid = np.stack([X, Y], axis=-1)

# potential_grid = potential(xy_grid)
# E_field_grid = -egrad(potential)(xy_grid)  # <--- evaluating the electric field

# import matplotlib.pyplot as plt
# plt.figure()
# plt.pcolormesh(X, Y, potential_grid)
# plt.quiver(X, Y, E_field_grid[..., 0], E_field_grid[..., 1])




