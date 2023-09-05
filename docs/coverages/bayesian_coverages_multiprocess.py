import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
from scipy.linalg import inv
from autograd.numpy import ndarray

from multiprocessing import Pool
from functools import partial
from typing import Tuple

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, GaussianPrior, Optimizer, MCMC, Prior
from debiased_spatial_whittle.bayes.funcs import transform, RW_MH, compute_hessian

np.random.seed(1535235325)

def f(i, grid: RectangularGrid, prior_mean: ndarray, prior_cov: ndarray):
    
    prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
    params = prior.sim()
    print(f'iteration: {i+1}, params={params.round(3)} \n')
    
    model = SquaredExponentialModel()   # TODO: try exponential model
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = nugget
    
    quantiles = [0.025,0.975]
    
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
        
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)
    dw.fit(x0=params, print_res=False)
    
    
    mle_niter= 1000
    mcmc_niter=5000
    acceptance_lag = mcmc_niter+1
    
    MLEs = dw.sim_MLEs(dw.res.x, niter=mle_niter, print_res=False)
    dw.compute_C3(dw.res.x)   # TODO: change C
    
    dw_mcmc = MCMC(dw, prior)
    post = dw_mcmc.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag)
    adj_post = dw_mcmc.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag, C=dw.C3)
    
    
    q     = np.quantile(post, quantiles, axis=0).T.flatten()
    q_adj = np.quantile(adj_post, quantiles, axis=0).T.flatten()
        
    probs     = np.sum(post < params, axis=0)/mcmc_niter
    probs_adj = np.sum(adj_post < params, axis=0)/mcmc_niter
    
    return params, q, q_adj, probs, probs_adj



def init_pool_processes():
    np.random.seed()
    # np.random.RandomState()
    # pass


n = (64, 64)
grid = RectangularGrid(n)

rho, sigma, nugget = 7., np.sqrt(1.), 0.1  # pick smaller rho
prior_mean = np.array([rho, sigma])    
prior_cov = np.array([[1., 0.], [0., .1]])  # TODO: PRIOR (VARIANCE) VERY IMPORTANT FOR COVERAGES/QUANTILES
prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
 
model = SquaredExponentialModel() 





quantiles = [0.025,0.975]
n_q = len(quantiles)
d=2 # len(prior_mean)



n_datasets=500
params_array = np.zeros((n_datasets,d))
dw_post_quants     = np.zeros((n_datasets,d*n_q))
adj_dw_post_quants = np.zeros((n_datasets,d*n_q))

dw_post_probs     = np.zeros((n_datasets, d))
adj_dw_post_probs = np.zeros((n_datasets, d))


g = partial(f, grid=grid, prior_mean=prior_mean, prior_cov=prior_cov)
with Pool(processes=20, initializer=init_pool_processes) as pool:

    for i, res in enumerate(pool.imap(g, range(n_datasets))):   # could do imap_unordered
        params, q, q_adj, probs, probs_adj = res
        
        params_array[i] = params
        dw_post_quants[i] = q
        adj_dw_post_quants[i] = q_adj
        
        dw_post_probs[i] = probs
        adj_dw_post_probs[i] = probs_adj
        
        print(q.round(3), q_adj.round(3), params.round(3), sep='\n')
        print('')
        
        print(probs.round(3), probs_adj.round(3), sep='\n')
        print('')
        


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False

prior_label = rf'$\rho \sim N({prior_mean[0]}, {np.diag(prior_cov)[0]})$, $\sigma \sim N({prior_mean[1]}, {np.diag(prior_cov)[1]})$ '

fig,ax = plt.subplots(2,2, figsize=(15,10))
fig.suptitle(f'Posterior quantile estimates, {n=}, {model.name}, {prior_label}', fontsize=24)#, fontweight='bold')
ax[0,0].hist(dw_post_probs[:,0], bins='sturges', edgecolor='k')
ax[0,1].hist(dw_post_probs[:,1], bins='sturges', edgecolor='k')

ax[1,0].hist(adj_dw_post_probs[:,0], bins='sturges', edgecolor='k')
ax[1,1].hist(adj_dw_post_probs[:,1], bins='sturges', edgecolor='k')

ax[1,0].set_xlabel( r'$\rho$', fontsize=22)
ax[1,1].set_xlabel( r'$\sigma$', fontsize=22)

# ax[0,0].text(.9, 240, 'debiased Whittle', color='r',fontsize=20)
# ax[1,0].text(.8, 160., 'Adjusted debiased Whittle', color='r',fontsize=20)

# for axis in ax.flatten():
    # axis.set_xticks([])
    # axis.set_yticks([])

fig.subplots_adjust(hspace=0.3, wspace=-1.5)
fig.tight_layout()
plt.show()

    
from scipy import stats
unif = stats.uniform(0,1)
qs = np.linspace(0,1, 1000)
theory_quants = unif.ppf(qs)

fig,ax = plt.subplots(1,2, figsize=(15,7))
fig.suptitle(f'Posterior quantiles QQ plot, {n=}, {model.name}, {prior_label}')
ax[0].plot(theory_quants, theory_quants, c='r', linewidth=3, label='standard uniform', zorder=10)
ax[0].plot(theory_quants, np.quantile(dw_post_probs[:,0], qs), 
           '.', c='g', markersize=10., label='dewhittle')
ax[0].plot(theory_quants, np.quantile(adj_dw_post_probs[:,0], qs), 
           '.', c='blue', markersize=10., label='adj dewhittle')
ax[0].legend()

ax[1].plot(theory_quants, theory_quants, c='r', linewidth=3, label='standard uniform', zorder=10)
ax[1].plot(theory_quants, np.quantile(dw_post_probs[:,1], qs), 
           '.', c='g', markersize=10., label='dewhittle')
ax[1].plot(theory_quants, np.quantile(adj_dw_post_probs[:,1], qs), 
           '.', c='blue', markersize=10., label='adj dewhittle')
ax[1].legend()


ax[0].set_xlabel( r'$\rho$', fontsize=22)
ax[1].set_xlabel( r'$\sigma$', fontsize=22)

fig.tight_layout()
plt.show()



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
                       dw_post_quants, adj_dw_post_quants,
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
