from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from abc import ABC, abstractmethod

from time import time
from autograd import grad, hessian
from numdifftools import Hessian
from autograd.scipy import stats
from autograd.numpy import ndarray
from scipy.optimize import minimize, basinhopping
from debiased_spatial_whittle.bayes.likelihoods_base import Likelihood
from debiased_spatial_whittle.bayes.prior import Prior
from debiased_spatial_whittle.bayes import GaussianPrior
from debiased_spatial_whittle.bayes.funcs import transform, compute_hessian

from typing import Union, Optional, Dict, Callable
from debiased_spatial_whittle.bayes.funcs import transform, fit


class Optimizer:
    
    def __init__(self, likelihood: Likelihood, transform_func: Optional[Callable] = None):
            
        self.likelihood = likelihood
        self._n_params = self.likelihood.n_params
        if transform_func is None:
            self.transform = self._transform
        else:
            self.transform = transform_func
    
    @property
    def n_params(self):
        return self._n_params
    
    def _transform(self, x: ndarray, inv:bool=True) -> ndarray:
        return x
    
    def fit(self, **kwargs):
        return fit(self.likelihood, **kwargs)
    
    
    # def fit(self, x0: Union[None, ndarray]=None, prior:bool = True, basin_hopping:bool = False,
    #         niter:int = 100, approx_grad:bool=False,
    #         print_res:bool = True, save_res:bool=True,
    #         loglik_kwargs: Optional[dict] =None, **optargs):
    #     '''
    #     optimize the log-likelihood function given the data
    #     includes optional global optimizer
    #     '''
    #     # TODO: clean this up        
    #     # TODO: separate class?
        
    #     if x0 is None: # TODO: fix this
    #         x0 = self.transform(np.ones(self.n_params), inv=False)
        
    #     if loglik_kwargs is None:
    #         loglik_kwargs = dict()
            
    #     attribute = 'MLE'
    #     def obj(x):     return -self.likelihood(x, **loglik_kwargs)  # TODO: assuming __call__ is loglike
            
    #     if not approx_grad:
    #         gradient = grad(obj)
    #     else:
    #         gradient = False
        
    #     if basin_hopping:          # for global optimization
    #     # TODO: separate method?
    #         minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient}
    #         res = basinhopping(obj, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, **optargs)
    #         success = res.lowest_optimization_result['success']
    #     else:            
    #         res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **optargs)
    #         success = res['success']
            
    #     if not success:
    #         print('Optimizer failed!')
    #         # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
            
    #     res['type'] = attribute
    #     # setattr(self, attribute, res)
        
    #     if attribute=='MLE':
    #         res['BIC'] = self.n_params * np.log(self.likelihood.grid.n_points) - 2*self.likelihood(res.x)         # negative logpost
        
    #     if print_res:
    #         print(f'{self.likelihood.label} {attribute}:  {self.transform(res.x).round(3)}')
            
    #     if save_res:
    #         setattr(self, 'res', res)
    #         setattr(self.likelihood, 'res', res)
            
    #     # self.propcov = self.compute_propcov(res.x, approx_grad)                      
    #     return res

class MCMC:
    
    def __init__(self, likelihood: Likelihood, prior: Prior):
        
        assert likelihood.n_params == prior.n_params, 'dimensions of likelihood and prior not equal'
    
        self.likelihood = likelihood
        self.prior = prior
        
        self._n_params = self.likelihood.n_params
    
    @property
    def n_params(self):
        return self._n_params
        
    def logpost(self, x: ndarray, **loglik_kwargs) -> float:
        return self.likelihood(x, **loglik_kwargs) + self.prior(x)
    
    def adj_logpost(self, x: ndarray, **loglik_kwargs) -> float:
        return self.likelihood.adj_loglik(x, **loglik_kwargs) + self.prior(x)
    
    
    # def fit(self, x0: Union[None, ndarray], prior:bool = True, basin_hopping:bool = False,
    #         niter:int = 100, approx_grad:bool=False,
    #         print_res:bool = True, save_res:bool=True,
    #         loglik_kwargs: Optional[dict] =None, **optargs):
    #     '''
    #     optimize the log-likelihood function given the data
    #     includes optional global optimizer
    #     '''
    #     # TODO: clean this up        
    #     # TODO: separate class?
        
    #     if x0 is None: # TODO: fix this
    #         x0 = np.zeros(self.n_params)
        
    #     if loglik_kwargs is None:
    #         loglik_kwargs = dict()
            
    #     if prior:                                         # for large samples, the prior is negligible
    #         attribute = 'MAP'
    #         def obj(x):     return -self.logpost(x, **loglik_kwargs)
    #     else:
    #         attribute = 'MLE'
    #         def obj(x):     return -self.likelihood(x, **loglik_kwargs)  # TODO: assuming __call__ is loglike
            
    #     if not approx_grad:
    #         gradient = grad(obj)
    #     else:
    #         gradient = False
        
    #     if basin_hopping:          # for global optimization
    #     # TODO: separate method?
    #         minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient}
    #         res = basinhopping(obj, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, **optargs)
    #         success = res.lowest_optimization_result['success']
    #     else:            
    #         res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **optargs)
    #         success = res['success']
            
    #     if not success:
    #         print('Optimizer failed!')
    #         # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
            
    #     res['type'] = attribute
    #     # setattr(self, attribute, res)
        
    #     if attribute=='MLE':
    #         res['BIC'] = self.n_params * np.log(self.likelihood.grid.n_points) - 2*self.likelihood(res.x)         # negative logpost
        
    #     if print_res:
    #         print(f'{self.likelihood.label} {attribute}:  {np.round(np.exp(res.x),3)}')
            
    #     if save_res:
    #         setattr(self, 'res', res)
            
    #     self.propcov = self.compute_propcov(res.x, approx_grad)                      
    #     return res
    
    def RW_MH(self, niter:int, adjusted:bool=False, acceptance_lag:int=1000, print_res:bool=True, **logpost_kwargs):
        '''Random walk Metropolis-Hastings: samples the specified posterior'''
        
        # TODO: mcmc diagnostics
        
        if adjusted:
            def posterior(x): return self.adj_logpost(x, **logpost_kwargs)
            label = 'adjusted ' + self.likelihood.label
            
        else:
            def posterior(x): return self.logpost(x, **logpost_kwargs)
            label = self.likelihood.label
            
        propcov   = compute_hessian(posterior, self.likelihood.res.x, inv=True)
            
        A     = np.zeros(niter, dtype=np.float64)
        U     = np.random.rand(niter)
            
        h = 2.38/np.sqrt(self.n_params)        
        props = h*np.random.multivariate_normal(np.zeros(self.n_params), propcov, size=niter)
        
        self.post_draws = np.zeros((niter, self.n_params))
        
        crnt_step = self.post_draws[0] = self.likelihood.res.x
        bottom    = posterior(crnt_step)
        # print(bottom)
        
        if print_res:
            print(f'{f"{label} MCMC":-^50}')
        t0 = time()
        for i in range(1, niter):
            
            prop_step = crnt_step + props[i]
            top       = posterior(prop_step)
            # print(top)
            
            A[i]      = np.min((1., np.exp(top-bottom)))
            if U[i] < A[i]:
                crnt_step  = prop_step
                bottom     = top
            
            self.post_draws[i]   = crnt_step
                
            if (i+1)%acceptance_lag==0:
                print(f'Iteration: {i+1}    Acceptance rate: {A[i-(acceptance_lag-1): (i+1)].mean().round(3)}    Time: {np.round(time()-t0,3)}s')
                
        # TODO: not returning A anymore!
        return self.post_draws


def test_mcmc():
    # TODO:  test this properly, not with DeWhittle
    from debiased_spatial_whittle.bayes import DeWhittle
    from debiased_spatial_whittle.grids import RectangularGrid
    from debiased_spatial_whittle.models import SquaredExponentialModel
    from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
    from debiased_spatial_whittle.plotting_funcs import plot_marginals
    
    from numpy.testing import assert_allclose
    
    model = SquaredExponentialModel()
    model.rho = 8
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid((64,64))
    
    params = np.array([8.,1.])
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    
    niter = 1000
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1)
    dw.fit(None, prior=False)
    # dewhittle_post, A = dw.RW_MH(niter)
    
    prior = GaussianPrior(np.zeros(2), np.eye(2)*100)
    mcmc = MCMC(dw, prior)
    mcmc.fit(None, prior=False)
    
    assert_allclose(dw.res.x, mcmc.res.x)
    assert_allclose(dw.propcov, mcmc.propcov)
    
    dewhittle_post = mcmc.RW_MH(1000, False)
    # title = 'posterior comparisons'
    # legend_labels = ['deWhittle']
    # plot_marginals([dewhittle_post], np.log(params), title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))

    
if __name__ == "__main__":
    test_mcmc()