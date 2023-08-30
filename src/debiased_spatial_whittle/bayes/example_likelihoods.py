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

np.random.seed(53252336)

inv = np.linalg.inv

model = ExponentialModel()
model.rho = 8
model.sigma = 1
model.nugget=0.1
grid = RectangularGrid((64,64))

params = np.array([8.,1.])
sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()


mle_niter=2000
dw = DeWhittle(z, grid, ExponentialModel(), nugget=0.1, transform_func=None)

dw.fit(x0=params)
MLEs = dw.sim_MLEs(dw.res.x, niter=mle_niter, print_res=False)
dw.compute_C_old(dw.res.x)
dw.compute_C2(dw.res.x)
dw.compute_C3(dw.res.x)

grads = dw.sim_J_matrix(dw.res.x, niter=5000)

p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
dw.update_model_params(dw.res.x)
H = d.fisher(dw.model, Parameters([ dw.model.rho, dw.model.sigma ]))
dw.H = H

dw.compute_C()

mcmc_niter=5000
dw_post = RW_MH(mcmc_niter, dw.res.x, dw, compute_hessian(dw, dw.res.x, inv=True))
adj_dw_post  = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C),  C=dw.C)
adj1_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C1), C=dw.C1)
adj2_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C2), C=dw.C2)
adj3_dw_post = RW_MH(mcmc_niter, dw.res.x, dw.adj_loglik, compute_hessian(dw.adj_loglik, dw.res.x, inv=True, C=dw.C3), C=dw.C3)

title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle', 'adj1 deWhittle', 'adj2 deWhittle', 'adj3 deWhittle']
posts = [dw_post, adj_dw_post, adj1_dw_post, adj2_dw_post, adj3_dw_post]
plot_marginals(posts, params, title, [r'$\rho$', r'$\sigma$'], legend_labels, shape=(1,2),  cmap='hsv')

