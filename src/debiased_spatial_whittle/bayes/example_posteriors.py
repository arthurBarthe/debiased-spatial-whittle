import autograd.numpy as np
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel, Parameters
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, MCMC, GaussianPrior, Optimizer
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.bayes.funcs import transform, RW_MH, compute_hessian, svd_decomp
from debiased_spatial_whittle.likelihood import DebiasedWhittle

from autograd import grad, hessian
inv = np.linalg.inv

np.random.seed(53252331)

n=(64,64)
grid = RectangularGrid(n)


rho, sigma, nugget = 7., 3., 0.1  # pick smaller rho
prior_mean = np.array([rho, sigma])    
prior_cov = np.array([[1., 0.], [0., .1]])  # TODO: PRIOR (VARIANCE) VERY IMPORTANT FOR COVERAGES/QUANTILES

prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
params = prior.sim()
print(f'params={params.round(3)}')

model = SquaredExponentialModel()   # TODO: try exponential model
model.rho = params[0]
model.sigma = params[1]
model.nugget = nugget


sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

# TODO: try on log-scale, transform_func!!!!
nsims = 2000
dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=None)
# dw.constant = 1/np.prod(n)

dw.fit(x0=params)
MLEs = dw.sim_MLEs(dw.res.x, niter=nsims, print_res=False)
dw.compute_C4(dw.res.x)
dw.compute_C5(dw.res.x)
dw.compute_C5_2(dw.res.x)

H = dw.fisher(dw.res.x)
grads = dw.sim_J_matrix(dw.res.x, niter=nsims)

dw.compute_C2()
dw.compute_C3(dw.res.x)
dw.compute_C6(dw.res.x)

mcmc_niter=10000
acceptance_lag = 1000

dw_post = MCMC(dw, prior)

dw_post_draws = dw_post.RW_MH(mcmc_niter, adjusted=False, acceptance_lag=acceptance_lag)
adj2_dw_post  = dw_post.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C2)
adj3_dw_post  = dw_post.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C3)
adj4_dw_post  = dw_post.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C4)
adj5_dw_post  = dw_post.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C5)
adj52_dw_post = dw_post.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C5_2)
adj6_dw_post = dw_post.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C6)

if False:
    gauss = Gaussian(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)
    gauss.fit(x0=params, included_prior=False, approx_grad=True)
    gauss_post  = RW_MH(mcmc_niter//5, gauss.res.x, gauss, compute_hessian(gauss, gauss.res.x, approx_grad=True, inv=True), acceptance_lag=100)
    

title = f'Posterior comparisons, {n=}'
density_labels = ['deWhittle', 'adj2 deWhittle', 'adj3 deWhittle', 'adj4 deWhittle', 'adj5 deWhittle', 'adj52 deWhittle', 'adj6 deWhittle']
posts = [dw_post_draws, adj2_dw_post, adj3_dw_post, adj4_dw_post, adj5_dw_post, adj52_dw_post, adj6_dw_post]
plot_marginals(posts, params, title, [r'$\rho$', r'$\sigma$'], density_labels, shape=(1,2),  cmap='hsv')

