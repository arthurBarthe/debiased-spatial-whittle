import numpy as np
from numpy import ndarray
from typing import Callable
from time import time
from scipy.stats import multivariate_normal


def RW_MH(niter: int,
          x0: ndarray,
          log_post: Callable,
          prop_cov: ndarray,
          acceptance_lag: int = 500,
          approx_grad: bool = False,
          **logpost_kwargs):
        
        '''Random walk Metropolis-Hastings: samples the specified posterior'''
        
        A = np.zeros(niter, dtype=np.float64)
        U = np.random.rand(niter)
        
        d = len(prop_cov)
        h = 2.38 / np.sqrt(d)       
        props = h * multivariate_normal(np.zeros(d),
                                        prop_cov).rvs(size=niter)
                                        
        
        post_draws = np.zeros((niter, d))
        
        crnt_step = post_draws[0] = x0
        bottom    = log_post(crnt_step, **logpost_kwargs)
        # print(bottom)
        
        print(f'{f" RW-MH MCMC":-^50}')
        t0 = time()
        for i in range(1, niter):
            
            prop_step = crnt_step + props[i]
            top       = log_post(prop_step, **logpost_kwargs)
            # print(top)
            
            A[i]      = np.min((1., np.exp(top-bottom)))
            if U[i] < A[i]:
                crnt_step  = prop_step
                bottom     = top
            
            post_draws[i]  = crnt_step
                
            if (i+1)%acceptance_lag==0:
                print(f'Iteration: {i+1}    Acceptance rate: {A[i-(acceptance_lag-1): (i+1)].mean().round(3)}    Time: {np.round(time()-t0,3)}s')
                
        return post_draws
