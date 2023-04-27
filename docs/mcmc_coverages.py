import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian



n = (64, 64)
rho, sigma, nugget = 10., np.sqrt(1.), 0.1

grid = RectangularGrid(n)
model = SquaredExponentialModel()
model.rho = rho
model.sigma = sigma
model.nugget = nugget


sampler = SamplerOnRectangularGrid(model, grid)

params = np.log([rho, sigma])

# stop


n_datasets=10
mcmc_niter=10000
mle_niter= 2000
acceptance_lag = mcmc_niter+1
d=len(params)

quantiles = [0.025, 0.975, 0.05, 0.95]
n_q = len(quantiles)

dewhittle_post_quantiles     = np.zeros((n_datasets,d*n_q))
adj_dewhittle_post_quantiles = np.zeros((n_datasets,d*n_q))
whittle_post_quantiles       = np.zeros((n_datasets,d*n_q))
adj_whittle_post_quantiles   = np.zeros((n_datasets,d*n_q))

print(f'True Params:  {np.round(np.exp(params),3)}')
for i in range(n_datasets):
    print(i+1, end=':\n')
    
    z = sampler()
    
    for likelihood in [DeWhittle]: # ,Whittle
            
        ll = likelihood(z, grid, SquaredExponentialModel(), nugget=nugget)
        ll.fit(None, prior=False, print_res=False)
        post, A = ll.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag)
        MLEs = ll.estimate_standard_errors_MLE(ll.res.x, monte_carlo=True, niter=mle_niter, print_res=False)
        ll.prepare_curvature_adjustment()
        adj_post, A = ll.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag)
    
        q     = np.quantile(post, quantiles, axis=0).T.flatten()
        q_adj = np.quantile(adj_post, quantiles, axis=0).T.flatten()
        if ll.label =='Debiased Whittle':
            dewhittle_post_quantiles[i] = q
            adj_dewhittle_post_quantiles[i] = q_adj
        else:
            whittle_post_quantiles[i] = q
            adj_whittle_post_quantiles[i] = q_adj
            
        print(q, q_adj, sep='\n')
        print('')

            
    
# stop
    
import pandas as pd    
post_list = ['dewhittle', 'adj_dewhittle', 'whittle', 'adj_whittle']
param_list = ['rho', 'sigma']
index  = pd.MultiIndex.from_product([post_list, param_list, quantiles], 
                                    names=["posterior", "parameter", "quantile"])

k = d*n_q*len(post_list)
posterior_quantiles = np.hstack((
                       dewhittle_post_quantiles, adj_dewhittle_post_quantiles,
                       whittle_post_quantiles, adj_whittle_post_quantiles,
                       )).reshape(n_datasets,k)


df = pd.DataFrame(posterior_quantiles, columns=index)

for i in range(k//2):
    cols = df.iloc[:, i*2:(i+1)*2]
    ll, param, q = cols.columns[0]
    interval = pd.arrays.IntervalArray.from_arrays(*cols.to_numpy().T, closed='both')
    if 'rho' == param:        
        count = interval.contains(params[0]).sum()
    else:
        count = interval.contains(params[1]).sum()
        
    print(f'{ll} coverage for parameter {param} at alpha={1-2*q}:   {count/n_datasets}')
