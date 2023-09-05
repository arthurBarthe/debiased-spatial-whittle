import autograd.numpy as np
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel, Parameters
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, MCMC, GaussianPrior, Optimizer
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.bayes.funcs import transform, RW_MH, compute_hessian
from debiased_spatial_whittle.likelihood import DebiasedWhittle

from autograd import grad, hessian

np.random.seed(53252335)

inv = np.linalg.inv

params = np.array([7.,1.])
model = SquaredExponentialModel()
model.rho = params[0]
model.sigma = params[1]
model.nugget=0.1

n=(64,64)
grid = RectangularGrid(n)

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()


mle_niter=2000
dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=None)

dw.fit(x0=params)
MLEs = dw.sim_MLEs(dw.res.x, niter=mle_niter, print_res=False)
dw.compute_C_old(dw.res.x)
dw.compute_C2(dw.res.x)
dw.compute_C3(dw.res.x)

H = dw.fisher(dw.res.x)
grads = dw.sim_J_matrix(dw.res.x, niter=5000)


dw.compute_C()
dw.compute_C4(dw.res.x)

mcmc_niter=5000
dw_post = RW_MH(mcmc_niter, dw.res.x, dw, compute_hessian(dw, dw.res.x, inv=True))
adj_dw_post  = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C),  C=dw.C)
adj1_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C1), C=dw.C1)
adj2_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C2), C=dw.C2)
adj3_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C3), C=dw.C3)
adj4_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C4), C=dw.C4)

gauss = Gaussian(z, grid, SquaredExponentialModel(), nugget=0.1)
gauss.fit(x0=params, included_prior=False, approx_grad=True)
gauss_post  = RW_MH(mcmc_niter//5, gauss.res.x, gauss, compute_hessian(gauss, gauss.res.x, approx_grad=True, inv=False), acceptance_lag=100)


title = 'posterior comparisons'
density_labels = ['gauss', 'deWhittle', 'adj deWhittle', 'adj1 deWhittle', 'adj2 deWhittle', 'adj3 deWhittle', 'adj4 deWhittle']
posts = [gauss_post, dw_post, adj_dw_post, adj1_dw_post, adj2_dw_post, adj3_dw_post, adj4_dw_post]
plot_marginals(posts, params, title, [r'$\rho$', r'$\sigma$'], density_labels, shape=(1,2),  cmap='hsv')

