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

np.random.seed(53252336)

inv = np.linalg.inv

params = np.array([7.,3.])
nugget = 0.1
model = SquaredExponentialModel()
model.rho = params[0]
model.sigma = params[1]
model.nugget=nugget

n=(64,64)
grid = RectangularGrid(n)
# taper = np.outer(np.blackman(n[0]), np.blackman(n[1]))

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

# TODO: try on log-scale, transform_func!!!!

mle_niter=2000
dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)

dw.fit(x0=params)
MLEs = dw.sim_MLEs(dw.res.x, niter=mle_niter, print_res=False)

dw.compute_C4(dw.res.x)
dw.compute_C5(dw.res.x)
dw.compute_C5_2(dw.res.x)

H = dw.fisher(dw.res.x)
grads = dw.sim_J_matrix(dw.res.x, niter=5000)


dw.compute_C2()
dw.compute_C3(dw.res.x)
dw.compute_C6(dw.res.x)

mcmc_niter=5000
dw_post = RW_MH(mcmc_niter, dw.res.x, dw, compute_hessian(dw, dw.res.x, inv=True))
adj2_dw_post  = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C2),  C=dw.C2)
adj3_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C3), C=dw.C3)
adj4_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C4), C=dw.C4)
adj5_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C5), C=dw.C5)
adj52_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C5_2), C=dw.C5_2)
adj6_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C6), C=dw.C6)

if False:
    gauss = Gaussian(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)
    gauss.fit(x0=params, included_prior=False, approx_grad=True)
    gauss_post  = RW_MH(mcmc_niter//5, gauss.res.x, gauss, compute_hessian(gauss, gauss.res.x, approx_grad=True, inv=True), acceptance_lag=100)
    

title = f'Posterior comparisons, {n=}'
density_labels = ['deWhittle', 'adj2 deWhittle', 'adj3 deWhittle', 'adj4 deWhittle', 'adj5 deWhittle', 'adj52 deWhittle', 'adj6 deWhittle']
posts = [dw_post, adj2_dw_post, adj3_dw_post, adj4_dw_post, adj5_dw_post, adj52_dw_post, adj6_dw_post]
plot_marginals(posts, params, title, [r'$\rho$', r'$\sigma$'], density_labels, shape=(1,2),  cmap='hsv')

