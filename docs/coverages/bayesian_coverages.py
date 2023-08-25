import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
from scipy.linalg import inv
from autograd.numpy import ndarray

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, GaussianPrior, Optimizer, MCMC

np.random.seed(1535235325)

n = (128, 128)
rho, sigma, nugget = 8., np.sqrt(1.), 0.1  # pick smaller rho

grid = RectangularGrid(n)

model = ExponentialModel()   # TODO: try sq. exponential model!!!
model.rho = rho
model.sigma = sigma
model.nugget = nugget


sampler = SamplerOnRectangularGrid(model, grid)
dw = DeWhittle(sampler(), grid, ExponentialModel(), nugget=nugget, transform_func=None) # just for initialization

prior_mean = np.array([rho, sigma])    
prior_cov = np.array([[1., 0.], [0., .01]])  # TODO: PRIOR (VARIANCE) VERY IMPORTANT FOR COVERAGES/QUANTILES

prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative


n_datasets=1000
mcmc_niter=5000
mle_niter= 500
acceptance_lag = mcmc_niter+1
d=len(prior_mean)


quantiles = [0.025,0.975]
n_q = len(quantiles)

dw_post_quants     = np.zeros((n_datasets,d*n_q))
adj_dw_post_quants = np.zeros((n_datasets,d*n_q))

dw_post_probs     = np.zeros((n_datasets, d))
adj_dw_post_probs = np.zeros((n_datasets, d))

params_array = prior.sim(n_datasets)
inside=0
for i, params in enumerate(params_array):
    print(f'iteration: {i+1}, params={params.round(3)}', end=':\n')
    
    z = dw.sim_z(params)
        
    dw = DeWhittle(z, grid, ExponentialModel(), nugget=nugget, transform_func=None)
        
    dw_opt = Optimizer(dw)
    dw_opt.fit()
    MLEs = dw.sim_MLEs(dw_opt.res.x, niter=mle_niter, print_res=False)
    dw.prepare_curvature_adjustment(dw_opt.res.x)
    
    dw_mcmc = MCMC(dw, prior)
    post = dw_mcmc.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag)
    adj_post = dw_mcmc.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag)

    q     = np.quantile(post, quantiles, axis=0).T.flatten()
    q_adj = np.quantile(adj_post, quantiles, axis=0).T.flatten()
    
    dw_post_quants[i] = q
    adj_dw_post_quants[i] = q_adj
    
    print(q.round(3), q_adj.round(3), params.round(3), sep='\n')
    print('')
    
    probs     = np.sum(post < params, axis=0)/mcmc_niter
    probs_adj = np.sum(adj_post < params, axis=0)/mcmc_niter
    
    dw_post_probs[i] = probs
    adj_dw_post_probs[i] = probs_adj
    
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

ax[0,0].text(.9, 240, 'debiased Whittle', color='r',fontsize=20)
ax[1,0].text(.8, 160., 'Adjusted debiased Whittle', color='r',fontsize=20)

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
fig.suptitle(f'QQ plot, posterior quantiles vs standard uniform, {n=}, {model.name}, {prior_label}')
ax[0].plot(theory_quants, theory_quants, c='r', linewidth=3, label='standard uniform', zorder=10)
ax[0].plot(theory_quants, np.quantile(dw_post_probs[:,0], qs), 
           '.', c='g', markersize=10., label='dewhittle')
ax[0].plot(theory_quants, np.quantile(adj_dw_post_probs[:,0], qs), 
           '.', c='blue', markersize=10., label='adj dewhittle')
ax[0].legend()
ax

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
