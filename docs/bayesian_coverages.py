import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv
from autograd.numpy import ndarray

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, GaussianPrior, MCMC



n = (64, 64)
rho, sigma, nugget = 8., np.sqrt(1.), 0.1  # pick smaller rho

grid = RectangularGrid(n)
# TODO: try exponential model
model = ExponentialModel()
model.rho = rho
model.sigma = sigma
model.nugget = nugget


sampler = SamplerOnRectangularGrid(model, grid)
dw = DeWhittle(sampler(), grid, ExponentialModel(), nugget=nugget) # just for initialization

prior_mean = np.log([rho, sigma])

prior = GaussianPrior(prior_mean, 0.1*np.eye(2))   # prior on unrestricted space


n_datasets=100
mcmc_niter=2000
mle_niter= 500
acceptance_lag = mcmc_niter+1
d=len(prior_mean)


quantiles = [0.025,0.975]
n_q = len(quantiles)

dewhittle_post_quantiles     = np.zeros((n_datasets,d*n_q))
adj_dewhittle_post_quantiles = np.zeros((n_datasets,d*n_q))

dewhittle_post_probs     = np.zeros((n_datasets, d))
adj_dewhittle_post_probs = np.zeros((n_datasets, d))

params_array = prior.sim(n_datasets)
inside=0
for i, params in enumerate(params_array):
    print(f'iteration: {i+1}, params={np.exp(params).round(3)}', end=':\n')
    
    z = dw.sim_z(np.exp(params))
        
    dw = DeWhittle(z, grid, ExponentialModel(), nugget=nugget)
    mcmc = MCMC(dw, prior)
    mcmc.fit(None, prior=False, print_res=False)
    post = mcmc.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag)
    MLEs = dw.sim_MLEs(np.exp(mcmc.res.x), niter=mle_niter, print_res=False)
    dw.fit(None, prior=False, print_res=False)
    dw.prepare_curvature_adjustment()
    adj_post = mcmc.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag)

    q     = np.quantile(post, quantiles, axis=0).T.flatten()
    q_adj = np.quantile(adj_post, quantiles, axis=0).T.flatten()
    
    dewhittle_post_quantiles[i] = q
    adj_dewhittle_post_quantiles[i] = q_adj
    
    print(q.round(3), q_adj.round(3), params.round(3), sep='\n')
    print('')
    
    probs     = np.sum(post < params, axis=0)/mcmc_niter
    probs_adj = np.sum(adj_post < params, axis=0)/mcmc_niter
    
    dewhittle_post_probs[i] = probs
    adj_dewhittle_post_probs[i] = probs_adj
    
    print(probs.round(3), probs_adj.round(3), sep='\n')
    print('')
    

            
    
stop
    
import pandas as pd    
post_list = ['dewhittle', 'adj_dewhittle', 'whittle', 'adj_whittle']
param_list = ['rho', 'sigma']
index  = pd.MultiIndex.from_product([post_list, param_list, quantiles], 
                          names=["posterior", "parameter", "quantile"])

new_index = []
for i,idx in enumerate(index):
    *post_params,q = idx
    new_index.append((*post_params,alphas_list[i],q))


k = d*n_q*len(post_list)
posterior_quantiles = np.hstack((
                       dewhittle_post_quantiles, adj_dewhittle_post_quantiles,
                       whittle_post_quantiles, adj_whittle_post_quantiles,
                       )).reshape(n_datasets,k)


df = pd.DataFrame(posterior_quantiles, columns=tuple(new_index))
df.columns.names = ["posterior", "parameter", "alpha", "quantile"]



# not a great solution
coverages={}
for idx in zip(new_index[::2], new_index[1::2]):
    
    cols = df[list(idx)]
    ll, param, alpha, q = cols.columns[0]
    interval = pd.arrays.IntervalArray.from_arrays(*cols.to_numpy().T, closed='both')
    if 'rho' == param:        
        count = interval.contains(params[0]).sum()
    else:
        count = interval.contains(params[1]).sum()
        
    print(f'{ll} coverage for parameter {param} at alpha={alpha}:   {count/n_datasets}')
    
    coverages[(ll,param,alpha)] = count/n_datasets
    # coverages.append(count/n_datasets)

coverages_arr = np.fromiter(coverages.values(), dtype=float)[None]
df_coverages = pd.DataFrame(coverages_arr,  columns=coverages.keys())

plt.plot(alphas, df_coverages['adj_dewhittle', 'rho'].to_numpy()[0], '.')
plt.show()
