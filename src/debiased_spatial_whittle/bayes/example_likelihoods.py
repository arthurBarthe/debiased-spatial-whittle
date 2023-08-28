import autograd.numpy as np
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel, SquaredModel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, MCMC, GaussianPrior, Optimizer
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid

from debiased_spatial_whittle.bayes.funcs import transform

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


prior_mean = np.array([model.rho.value, model.sigma.value])

prior = GaussianPrior(prior_mean, 0.1*np.eye(2))   # prior on unrestricted space

mle_niter=500
dw = DeWhittle(z, grid, ExponentialModel(), nugget=0.1, transform_func=None)

dw.fit()
MLEs = dw.sim_MLEs(dw.res.x, niter=mle_niter, print_res=False)
dw.compute_C_old(dw.res.x)
dw.compute_C2(dw.res.x)
dw.compute_C3(dw.res.x)

mcmc_niter=5000
dw_mcmc = MCMC(dw, prior)
dw_post = dw_mcmc.RW_MH(mcmc_niter, acceptance_lag=1000)
adj_dw_post = dw_mcmc.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=1000, C=dw.C)
adj2_dw_post = dw_mcmc.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=1000, C=dw.C2)
adj3_dw_post = dw_mcmc.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=1000, C=dw.C3)

title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle', 'adj2 deWhittle', 'adj3 deWhittle']
posts = [dw_post, adj_dw_post, adj2_dw_post, adj3_dw_post]
plot_marginals(posts, params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2),  cmap='hsv')


stop


grads = dw.sim_var_grad(np.exp(dw_mcmc.res.x), niter=mle_niter, print_res=True)

HJH1 = dw.MLEs_cov   # J^-1

H = -np.mean(dw.hess_at_params, axis=0)
# H = -hessian(dw)(dw_mcmc.res.x)

print(HJH1)
print(inv(H) @ dw.J @ inv(H))
# print(H @ inv(dw.J) @ H)


