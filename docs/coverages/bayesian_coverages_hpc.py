#!/usr/bin/env python3


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
from debiased_spatial_whittle.bayes import Likelihood, DeWhittle, Whittle, Gaussian, GaussianPrior, Optimizer, MCMC, Prior
from debiased_spatial_whittle.bayes.funcs import transform, RW_MH, compute_hessian

# np.random.seed(1535235325)

import os, sys

start = 1
n_datasets= 100

def get_last_line_number(file_name: str):
    ''' 
    This will open the file "primes.txt" and try to convert the last
    line to an integer. This should be the last prime that was found. 
    Return that prime. 
    '''
    try:
        with open(file_name, 'r') as f:
            last_line = f.readlines()[-1]
            last_line_number = int( last_line.split(' ')[0] ) + 1
            print(last_line_number)
    except IOError:
        print('Error: Could not open existing file.')
        sys.exit(1)
    except ValueError:  
        print('Error: Last line could not be converted to an integer.') 
        sys.exit(1)

    return last_line_number



def func(i, likelihood: Likelihood, grid: RectangularGrid, prior_mean: ndarray, prior_cov: ndarray):
    
    prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
    params = prior.sim()
    print(f'iteration: {i+1}, params={params.round(3)} \n')
    
    nugget=0.1
    model = SquaredExponentialModel()   # TODO: try exponential model
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = nugget
    
    quantiles = [0.025,0.975]
    
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
        
    dw = likelihood(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)
    dw.fit(x0=params, print_res=False)
    # TODO change
    
    mle_niter  = 100
    mcmc_niter = 500
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
    # pass


def main():
    
    global start, n_datasets
    
    n = (64, 64)
    grid = RectangularGrid(n)
    
    rho, sigma, nugget = 7., np.sqrt(1.), 0.1  # pick smaller rho
    prior_mean = np.array([rho, sigma])    
    prior_cov = np.array([[1., 0.], [0., .1]])  # TODO: PRIOR (VARIANCE) VERY IMPORTANT FOR COVERAGES/QUANTILES
    # prior = GaussianPrior(prior_mean, prior_cov)   # make sure sigma not negative
     
    model = SquaredExponentialModel() 
    
    likelihood = DeWhittle
    
    file_name = f'{likelihood.__name__}_{n[0]}x{n[1]}_{model.name}.txt'
    
    with open('submit_coverages.sh', 'r') as f:
        text = f.read()
        idx = text.find('ncpus=')
        ncpus = int(text[ idx+6 : idx+8 ])   # TODO: regex?? better way, +9 if >99!!
        
    print(ncpus)
    
    g = partial(func, likelihood=likelihood, grid=grid, prior_mean=prior_mean, prior_cov=prior_cov)
    with Pool(processes=None, initializer=init_pool_processes, maxtasksperchild=1) as pool:
    
        for i, result in enumerate(pool.imap(g, range(n_datasets))):   # could do imap_unordered
                
            params, q, q_adj, probs, probs_adj = result
                        
            print(q.round(3), q_adj.round(3), params.round(3), sep='\n')
            print('')
            
            print(probs.round(3), probs_adj.round(3), sep='\n')
            print('')
            
            print(file_name)
            if os.path.exists(file_name):
                # start = get_last_line_number(file_name)
                new_file = False
                print('Found an existing file')
                
            else:
                new_file = True

            if start >= n_datasets:
                print('Already finished.')
                sys.exit(1)
            
            with open(file_name, 'a+') as f:
                if new_file:
                    f.write(f'# prior_mean={prior_mean.tolist()}, prior_cov={prior_cov.tolist()}\n')
                    f.write('quantile adj_quantile prob prob_adj\n')
                
                f.write(f'{q} {q_adj} {probs} {probs_adj}\n')
                

if __name__ == '__main__':
    main()