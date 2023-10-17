#!/usr/bin/env python3


import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
from scipy.linalg import inv
from autograd.numpy import ndarray

import multiprocessing as mp
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
from debiased_spatial_whittle.bayes import Likelihood, DeWhittle, Whittle, Gaussian, GaussianPrior, Optimizer, MCMC, Prior
from debiased_spatial_whittle.bayes.funcs import transform, RW_MH, compute_hessian

# np.random.seed(1535235325)

import os, sys
import matplotlib.pyplot as plt

# start = 1
# n_datasets= 500


def func(i: int, 
         likelihood: Likelihood, 
         grid: RectangularGrid, 
         prior_mean: ndarray, 
         prior_cov: ndarray,
         mcmc_niter: int,
         mle_niter: int):
    
    prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
    params = prior.sim()
    print(f'iteration: {i+1}, params={params.round(3)} \n')
    
    nugget=0.1
    model = SquaredExponentialModel()   # TODO: try exponential model
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = nugget
    
    quantiles = [0.025, 0.975, 0.4, 0.6, 0.5]
    
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
        
    name = likelihood.__name__
    approx_grad = True if name=='Gaussian' else False
    
    ll = likelihood(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)   # TODO: use params on logspace!!
    ll.fit(x0=params, print_res=False, approx_grad = approx_grad)
    
    # MCMC
    burnin = 1000
    mcmc = MCMC(ll, prior)
    acceptance_lag = mcmc_niter+1
    post = mcmc.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag, approx_grad=approx_grad)[burnin:]
        
    if name in {'DeWhittle', 'Whittle'}:
        
        # MLEs = ll.sim_MLEs(ll.res.x, niter=mle_niter, print_res=False)
        H = ll.fisher(ll.res.x)
        ll.sim_J_matrix(ll.res.x, niter=mle_niter)
        ll.compute_C3(ll.res.x)   # TODO: change C # TODO: for C3 can get singular matrix error!!
        
        adj_post = mcmc.RW_MH(mcmc_niter, adjusted=True, 
                              acceptance_lag=acceptance_lag, C=ll.C3)[burnin:]
    else:
        adj_post = np.zeros((mcmc_niter, ll.n_params))
        
    # plot_marginals([post, adj_post], truths=params)

    q     = np.quantile(post, quantiles, axis=0).T.flatten()
    q_adj = np.quantile(adj_post, quantiles, axis=0).T.flatten()

    probs     = np.sum(post < params, axis=0)/mcmc_niter
    probs_adj = np.sum(adj_post < params, axis=0)/mcmc_niter
    
    return params, q, q_adj, probs, probs_adj



def init_pool_processes():
    np.random.seed()
    # pass


def main():
    
    # global start, n_datasets
    
    # likelihood = DeWhittle
    # n1 = 64
    # n_datasets = 10
    
    likelihood_name, n1, n_datasets = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])    # argument for C3? # TODO: make args string?
    if likelihood_name in {'Gaussian', 'DeWhittle', 'Whittle'}:
        likelihood = eval(likelihood_name)
        
    n = (n1,n1)
        
    print(f'{sys.argv[0]} run with likelihood={likelihood.__name__} with {n=} and {n_datasets=}')
    
    grid = RectangularGrid(n)
    
    rho, sigma, nugget = 3., 3., 0.1  # pick smaller rho
    prior_mean = np.array([rho, sigma])    
    prior_cov = np.array([[0.25, 0.], [0., .1]])  # TODO: PRIOR (VARIANCE) VERY IMPORTANT FOR COVERAGES/QUANTILES
    # prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
     
    model = SquaredExponentialModel() 
        
    file_name = f'{likelihood.__name__}_{n[0]}x{n[1]}_{model.name}.txt'
    
    mle_niter  = 2000
    mcmc_niter = 10000

    g = partial(func, 
                likelihood=likelihood, 
                grid=grid, 
                prior_mean=prior_mean, 
                prior_cov=prior_cov,
                mcmc_niter=mcmc_niter,
                mle_niter=mle_niter)
    
    
    with open('submit_coverages.sh', 'r') as f:
        text = f.read()
        idx = text.find('ncpus=')
        nprocesses = int(text[ idx+6 : idx+8 ]) - 2  # TODO: change when ncpus<100!! -2 to not overload error
    
    nprocesses = 25  # mp.cpu_count()
    
    with Pool(processes=nprocesses, initializer=init_pool_processes, maxtasksperchild=1) as pool:
    
        for i, result in enumerate(pool.imap(g, range(n_datasets))):   # could do imap_unordered
                
            params, q, q_adj, probs, probs_adj = result   # TODO: save parameters as well!!!
                        
            print(params.round(3), q.round(3), q_adj.round(3), sep='\n')
            print('')
            
            print(probs.round(3), probs_adj.round(3), sep='\n')
            print('')
            
            print(i, file_name)
            if os.path.exists(file_name):
                # start = get_last_line_number(file_name)
                new_file = False
                print('Found an existing file')
                
            else:
                new_file = True
            
            with open(file_name, 'a+') as f:
                if new_file:
                    f.write(f'# prior_mean={prior_mean.tolist()}, prior_cov={prior_cov.tolist()}\n')
                    f.write('parameters quantiles adj_quantiles probs prob_adjs\n')
                
                res = np.concatenate([params, q, q_adj, probs, probs_adj])
                for i, number in enumerate(res):
                    val = f'{number:f}'
                    if i+1<len(res):
                        f.write(val + ' ')
                    else:
                        f.write(val + '\n')
                

if __name__ == '__main__':
    main()
