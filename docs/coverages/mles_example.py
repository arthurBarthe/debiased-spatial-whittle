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
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian, GaussianPrior, Optimizer, MCMC, Prior
from debiased_spatial_whittle.bayes.funcs import transform, RW_MH, compute_hessian

# np.random.seed(1535235325)

import os
import sys

start=1
n_sims= 2000



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



def main():
    
    global start, n_sims 
    
    n = (64, 64)
    grid = RectangularGrid(n)

    params = np.array([7.,1.])
    nugget=0.1

    model = SquaredExponentialModel()   # TODO: try exponential model
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = nugget


    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
        
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)
    dw.fit(x0=params, print_res=True)
    
    # Check if an existing list of primes exists. 
    file_name = f'MLEs_{n=}_{model.name}_{params=}.txt'.replace(' ','')
    print(file_name)
    if os.path.exists(file_name):
        start = get_last_line_number(file_name)
        print('Found an existing file')

    if start >= n_sims:
        print('Already finished.')
        sys.exit(1)
       
    # Open the output file for appending.
    with open(file_name, 'a+') as f:
        
        # simulation
        for i in range(start, n_sims+1):
            z = sampler()
                
            dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None)
            dw.fit(x0=params, print_res=True)
            f.write(f'{i} {dw.res.x}\n')
    
if __name__ == '__main__':
    main()
