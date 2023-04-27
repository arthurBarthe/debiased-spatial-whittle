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


ndatasets=500
mcmc_niter=1000
mle_niter= 100
acceptance_lag = mcmc_niter+1
d=len(params)
deWhittle_post_q5   = np.zeros((ndatasets,d,2))
deWhittle_post_q10  = np.zeros((ndatasets,d,2))
adj_deWhittle_post_q5   = np.zeros((ndatasets,d,2))
adj_deWhittle_post_q10  = np.zeros((ndatasets,d,2))
Whittle_post_q5   = np.zeros((ndatasets,d,2))
Whittle_post_q10  = np.zeros((ndatasets,d,2))
adj_Whittle_post_q5   = np.zeros((ndatasets,d,2))
adj_Whittle_post_q10  = np.zeros((ndatasets,d,2))
print(f'True Params:  {np.round(np.exp(params),3)}')
for i in range(ndatasets):
    print(i+1, end=': ')
    
    z = sampler()
    
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=nugget)

    dw.fit(None, prior=False, print_res=False)
    dewhittle_post, A = dw.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag)
    MLEs = dw.estimate_standard_errors_MLE(dw.res.x, monte_carlo=True, niter=mle_niter, print_res=False)
    dw.prepare_curvature_adjustment()
    adj_dewhittle_post, A = dw.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag)
    
    
    q5, q10 = np.quantile(dewhittle_post, [[0.025, 0.975],[0.05, 0.95]], axis=0)
    deWhittle_post_q5[i]  = q5.T
    deWhittle_post_q10[i] = q10.T
    
    adj_q5, adj_q10 = np.quantile(adj_dewhittle_post, [[0.025, 0.975],[0.05, 0.95]], axis=0)
    adj_deWhittle_post_q5[i]  = adj_q5.T
    adj_deWhittle_post_q10[i] = adj_q10.T
    print(q5.T, adj_q5.T, sep='\n')
    
    whittle = Whittle(z, grid, SquaredExponentialModel(), nugget=nugget)
    whittle.fit(None, False)
    whittle_post, A = whittle.RW_MH(mcmc_niter)
    whittle.estimate_standard_errors_MLE(whittle.res.x, monte_carlo=True, niter=mle_niter, print_res=False)
    whittle.prepare_curvature_adjustment()
    adj_whittle_post, A = whittle.RW_MH(mcmc_niter, adjusted=True)
    
    q5, q10 = np.quantile(whittle_post, [[0.025, 0.975],[0.05, 0.95]], axis=0)
    Whittle_post_q5[i]  = q5.T
    Whittle_post_q10[i] = q10.T
    
    adj_q5, adj_q10 = np.quantile(adj_whittle_post, [[0.025, 0.975],[0.05, 0.95]], axis=0)
    adj_Whittle_post_q5[i]  = adj_q5.T
    adj_Whittle_post_q10[i] = adj_q10.T
    print(q5.T, adj_q5.T, sep='\n')
    
    print('')
    
    

    
import pandas as pd    
post_list = ['dewhittle', 'adj_dewhittle', 'whittle', 'adj_whittle']
param_list = ['rho', 'sigma']
index  = pd.MultiIndex.from_product([post_list, param_list, [2.5, 97.5, 5, 95]], 
                                    names=["posterior", "parameter", "quantile"])

k = 4*d*len(post_list)
quantiles = np.hstack((deWhittle_post_q5, deWhittle_post_q10,
                       adj_deWhittle_post_q5, adj_deWhittle_post_q10,
                       Whittle_post_q5, Whittle_post_q10,
                       adj_Whittle_post_q5, adj_Whittle_post_q10
                       )).reshape(500,k)

# messed up the column order
idx = np.array([0,1,4,5,2,3,6,7])
columns = np.r_[idx,idx+8,idx+16, idx+32]
posterior_quantiles = quantiles[:,columns]

df = pd.DataFrame(posterior_quantiles, columns=index)

for i in range(32//2):
    cols = df.iloc[:, i*2:(i+1)*2]
    ll, param, q = cols.columns[0]
    interval = pd.arrays.IntervalArray.from_arrays(*cols.to_numpy().T, closed='both')
    if 'rho' == param:        
        count = interval.contains(params[0]).sum()
    else:
        count = interval.contains(params[1]).sum()
        
    print(f'{ll} coverage, for parameter {param}, at q{100-2*q} = {count/ndatasets}')
