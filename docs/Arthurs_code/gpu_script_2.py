import torch
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())
DEVICE = 'cuda:0'

torch.manual_seed(1712)

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('torch')
BackendManager.device = DEVICE

from debiased_spatial_whittle.models import Parameters

import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternCovarianceModel, CovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid, Grid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

from typing import Callable, List
from numpy import ndarray
from numdifftools import Hessian
from scipy.stats import multivariate_normal

from torch.linalg import cholesky, inv, matmul
from debiased_spatial_whittle.models import Parameters

from utils import RW_MH

def make_filename(model, prior, grid_size, adjusted, C: str):
    prior_info = model.name + str(model.nugget.value)
    prior_info += '__(' + str(prior.mean[0]) + ',' + str(prior.cov[0, 0]) + ')'
    prior_info += '__(' + str(prior.mean[1]) + ',' + str(prior.cov[1, 1]) + ')'
    return '__'.join([prior_info, str(grid_size), str(adjusted), str(C)])



model = SquaredExponentialModel()
model.nugget = 0.05

est_params = Parameters((model.rho, model.sigma))
model.rho = 14
model.sigma = 2

shape = (1000, 1000)
mask_france = grids.ImgGrid(shape).get_new()
grid_france = RectangularGrid(shape)
grid_france.mask = torch.ones(shape, device=DEVICE)
# grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)
sampler.n_sims = 50

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
options = dict(ftol=np.finfo(float).eps * 10, gtol=1e-20)  # for extreme precision, ftol is stops iteration, gtol is for gradient
estimator = Estimator(debiased_whittle, use_gradients=True, optim_options=options)




prior = multivariate_normal([12., 1.2], np.array([[1.5, 0.], [0., 0.25]]))
ADJUSTED = True
C = 'C5'

# number of independent samples
n_datasets = 250
# number of mcmc steps
n_mcmc = 4000
# number of mle samples
n_sims = 100

filename = make_filename(model,prior, grid_france.n, ADJUSTED, C)
print(filename)

try:
    data = np.loadtxt(filename)
    qs = list(data)
except:
    qs = []


def sim_MLEs(rho: float,
             sigma: float,
             model: CovarianceModel,
             grid: Grid,
             likelihood: DebiasedWhittle,   # do we really need extreem precision?
             nsims: int = 1000,
             print_res : bool = True):
    
    
    opt_options = dict(ftol=np.finfo(float).eps * 1e7, gtol=1e-10)   # moderate accuracy when simulating mles
    estimator = Estimator(likelihood, use_gradients=False, optim_options=opt_options)
    # print(estimator.optim_options)
    
    model.rho, model.sigma = rho, sigma
    sampler = SamplerOnRectangularGrid(model, grid)
    
    mles = []
    for i in range(nsims):
        
        z = sampler()
        
        # compute dbw at true parameter for later check
        dbw_0 = debiased_whittle(z, model)
        model.rho, model.sigma = None, None

        estimate = estimator(model, z, x0 = np.array([rho, sigma]))     # optimizes log-likelihood
        # print(estimator.opt_result.hess_inv.todense())
        
        dbw_hat = debiased_whittle(z, estimate)
        if dbw_0 < dbw_hat:
            print('Optimization failed')
            continue
        
        rho_hat, sigma_hat = estimate.rho.value, estimate.sigma.value
        mle = np.array([rho_hat, sigma_hat])
        
        if print_res:
            print(f'{i+1})   MLE:  {mle.round(3)}')

        mles.append(mle)
    
    return np.array(mles)

def compute_C5(mles: ndarray, hessian_inv: ndarray):
    '''Compute un-normalized C5 adjustment matrix'''
    B = np.linalg.cholesky(np.cov(mles.T))              # inv(H) J inv(H)
    
    H_inv = hessian_inv.copy()                          # inverse Hessian, inv of observed fisher
    L     = np.linalg.cholesky(H_inv)                       
    C5    = L @ np.linalg.inv(B)
    return torch.tensor(C5)    # this is un-normalized C5!
        

# TODO: check is computing periodogram every iteration of dwb!!

for i_dataset in range(n_datasets):
    # sample parameters from prior
    theta0 = prior.rvs()
    model.rho, model.sigma = theta0
    print(f'Dataset #{i_dataset+1} \t theta_0 = {theta0.round(3)}')
    
    # sample random field
    sampler = SamplerOnRectangularGrid(model, grid_france)
    z = sampler()
    # compute dbw at true parameter for later check
    dbw_0 = debiased_whittle(z, model)
    # point estimate
    model.rho, model.sigma = None, None
    estimate = estimator(model, z, x0 = theta0)
    dbw_hat = debiased_whittle(z, estimate)
    if dbw_0 < dbw_hat:
        print('Optimization failed')
        continue
    
    rho_hat, sigma_hat = estimate.rho.value, estimate.sigma.value
    theta_hat = torch.tensor([rho_hat, sigma_hat]).reshape((-1, 1))
    MAP = torch.tensor([rho_hat, sigma_hat])
    print(f'MAP={MAP.numpy().round(3)}')
    
    
    if ADJUSTED:
        # Compute adjustment matrix
        constant = np.sqrt(grid_france.n_points / 2)
        if C == 'C2':
            params_for_gradient = Parameters((estimate.rho, estimate.sigma))
            H    = debiased_whittle.fisher(estimate, params_for_gradient)
            Jhat = debiased_whittle.jmatrix_sample(estimate, params_for_gradient, n_sims=n_sims, block_size=1)
            M_A  = cholesky(matmul(H, matmul(inv(Jhat), H))).T
            M    = cholesky(H).T
            C2   = matmul(inv(M), M_A) / constant
        
            lhs = C2.T @ (1 / 2 * H  * grid_france.n_points) @ C2
            if not torch.allclose(lhs, H @ inv(Jhat) @ H):
                continue
            
        elif C == 'C5':
            mles = sim_MLEs(rho_hat, sigma_hat, model, grid_france,  debiased_whittle,  nsims=n_sims)
            hessian_inv = estimator.opt_result.hess_inv.todense()     
            C5 = compute_C5(mles, hessian_inv)
            
            lhs = C5.T @ np.linalg.inv(hessian_inv) @ C5
            if not np.allclose(lhs, np.linalg.inv(np.cov(np.array(mles).T))):
                continue
            
            C5 /= constant
    
    # define posterior
    def log_posterior(rho, sigma, **kwargs):
        model.rho, model.sigma = rho, sigma
        return -debiased_whittle(z, model) / 2 * grid_france.n_points + prior.logpdf([rho, sigma])
    
    def log_posterior_adjusted(rho, sigma, C):
        u = torch.tensor([rho, sigma]).reshape((-1, 1))
        theta_star = theta_hat + matmul(C.cpu(), u - theta_hat)
        model.rho, model.sigma = theta_star[0].item(), theta_star[1].item()
        return -debiased_whittle(z, model) / 2 * grid_france.n_points + prior.logpdf([rho, sigma])
        
    if ADJUSTED:
        log_posterior = log_posterior_adjusted
    
    log_post = lambda x: log_posterior(x[0], x[1], C = eval(C))
    prop_cov = np.linalg.inv(-Hessian(log_post)(MAP))
    
    # run mcmc
    posterior_sample = RW_MH(n_mcmc, MAP, log_post, prop_cov)
    
    q = np.mean(theta0 <= posterior_sample[n_mcmc // 2:, ], axis=0)
    print(f'{q=}')
    qs.append(q)
    with open(filename, 'a') as f:
        np.savetxt(f, q.reshape((1, -1)))
    # plot
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    ax[0].plot(posterior_sample[:,0])
    ax[0].axhline(theta0[0], c='r')
    ax[1].plot(posterior_sample[:,1])
    ax[1].axhline(theta0[1], c='r')
    quantiles = np.quantile(np.array(qs), np.linspace(0, 1, 100), axis=0)
    ax[2].plot(np.linspace(0, 1, 100), quantiles)
    ax[2].plot([0, 1], [0, 1], c='k')
    ax[2].legend(('rho', 'sigma'), fontsize=16)
    fig.tight_layout()
    plt.show()
