from typing import Optional

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

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, TSamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.models import CovarianceModel

from typing import Union

fftn = np.fft.fftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


class Likelihood(ABC):
    
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, nugget: Optional[float] =0.1, use_taper: Union[None, ndarray]=None):
        
        self._z = z
        self.grid = grid
        # TODO: make this property
        self.n = self.grid.n
        
        self.use_taper = use_taper
        self.periodogram = Periodogram(taper=use_taper)
        self._I = self.periodogram(z)   # TODO: make periodogram cached_property
        
        # TODO: add model args
        self.model = model
        if self.model.name == 'TMultivariateModel':   # TODO: better way to do this?
            self.sampler_model = TSamplerOnRectangularGrid
            self.model.nugget_0 = nugget
            
        elif self.model.name == 'SquaredSamplerOnRectangularGrid': # TODO: this is not name of model
            self.sampler_model = SquaredSamplerOnRectangularGrid
            self.model.nugget = nugget
            
        else:
            self.sampler_model = SamplerOnRectangularGrid
            self.model.nugget = nugget
        
        self._free_params = model.free_params
        self._n_params = len(self._free_params)
        
    
    @property
    def z(self):
        return self._z
    
    @property
    def I(self):
        return self._I
        
    @property
    def n_params(self):
        """Number of parameters of the model"""
        return self._n_params
    
    @property
    def free_params(self):
        return self._free_params
    
    @property
    def n_points(self):
        return self.grid.n_points
    
    @abstractmethod
    def update_model_params(self, params: ndarray) -> None:
       all_params = self.model.params
       updates = dict(zip(all_params.names, params))
       all_params.update_values(updates)
       return
    
    @abstractmethod
    def __call__(self, x: ndarray) -> float: # loglik
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @property
    def label(self) -> str:
        pass
    
    @abstractmethod
    def cov_func(self, params: ndarray, lags: Optional[list[ndarray, ...]] =None, **cov_args) -> list[ndarray, ...]:
        '''compute covariance func on a grid of lags given parameters'''
        
        # TODO: only for dewhittle and whittle
        if lags is None:
            lags = self.grid.lags_unique

        self.update_model_params(params)
        return ifftshift(self.model(np.stack(lags), **cov_args))
    
    
    @abstractmethod
    def mymethod(self,x:Union[int, ndarray]):
        self.abc = x
        return x+5
    
    
    @abstractmethod
    def logprior(self, x: ndarray) -> float:
        '''uninformative prior on the transformed (unrestricted space)'''
        k = self.n_params
        return stats.multivariate_normal.logpdf(x, np.zeros(k), cov=np.eye(k)*100)


    @abstractmethod
    def logpost(self, x: ndarray, **loglik_kwargs) -> float:
        return self(x, **loglik_kwargs) + self.logprior(x)

    
    @abstractmethod
    def adj_loglik(self, x: ndarray, **loglikargs) -> float: 
        return self(self.res.x + self.C @ (x - self.res.x), **loglikargs)

    @abstractmethod
    def adj_logpost(self, x: ndarray) -> float:
        return self.adj_loglik(x) + self.logprior(x)

    
    @abstractmethod
    def fit(self, x0: Union[None, ndarray], prior:bool = True, basin_hopping:bool = False,
            niter:int = 100, approx_grad:bool=False,
            print_res:bool = True, save_res:bool=True,
            loglik_kwargs: Optional[dict] =None, **optargs):
        '''
        optimize the log-likelihood function given the data
        includes optional global optimizer
        '''
        
        # TODO: separate class?
        if x0 is None:
            x0 = np.zeros(self.n_params)
        
        if loglik_kwargs is None:
            loglik_kwargs = dict()
            
        if prior:                                         # for large samples, the prior is negligible
            attribute = 'MAP'
            def obj(x):     return -self.logpost(x, **loglik_kwargs)
        else:
            attribute = 'MLE'
            def obj(x):     return -self(x, **loglik_kwargs)
            
        if not approx_grad:
            gradient = grad(obj)
        else:
            gradient = False
            
        if basin_hopping:          # for global optimization
            minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient}
            res = basinhopping(obj, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, **optargs)
            success = res.lowest_optimization_result['success']
        else:            
            res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **optargs)
            success = res['success']
            
        if not success:
            print('Optimizer failed!')
            # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
            
        res['type'] = attribute
        # setattr(self, attribute, res)
        
        if attribute=='MLE':
            res['BIC'] = self.n_params * np.log(self.grid.n_points) - 2*self(res.x)         # negative logpost
        
        if print_res:
            print(f'{self.label} {attribute}:  {np.round(np.exp(res.x),3)}')
            
        if save_res:
            setattr(self, 'res', res)
        
            if approx_grad:
                hess = Hessian(self.logpost)(self.res.x)
                self.propcov = -np.linalg.inv(hess)
            else:    
                try:
                    # TODO: clean this up
                    self.propcov = np.linalg.inv(-hessian(self.logpost)(self.res.x))
                    
                    if not np.all(np.isfinite(self.propcov)):       # use numerical diff
                        hess = Hessian(self.logpost)(self.res.x)
                        self.propcov = -np.linalg.inv(hess)
                        
                except np.linalg.LinAlgError:
                    print('Singular propcov')
                    self.propcov = False
                        
        return res
    
    
    @abstractmethod
    def sim_z(self, params: Union[None, ndarray]=None):
        if params is None:
            params = np.exp(self.res.x)
            
        self.update_model_params(params)
        sampler = self.sampler_model(self.model, self.grid)
        # print(sampler.gaussian_sampler.sampling_grid.n)
        return sampler()
    
    @abstractmethod
    def sim_MLEs(self, params: ndarray, niter:int=5000, print_res:bool=True, **fit_kwargs) -> ndarray:
        
        i = 0
        self.MLEs = np.zeros((niter, self.n_params), dtype=np.float64)
        while niter>i:
            
            _z = self.sim_z(params)

            if False:
                # print(_z[4,11])
                import matplotlib.pyplot as plt
                plt.imshow(_z, origin='lower')
                plt.show()
                
            loglik_kwargs = {'z':_z}
            res = self.fit(x0=np.log(params), prior=False, print_res=False, 
                                                   save_res=False,
                                                   loglik_kwargs=loglik_kwargs,
                                                   **fit_kwargs)
            
            if not res['success']:
                continue
            else:
                if print_res:
                    print(f'{i+1})   MLE:  {np.round(np.exp(res.x),3)}')
                self.MLEs[i] = res.x
                i+=1
            
        self.MLEs_cov = np.cov(self.MLEs.T)
        return self.MLEs


    @abstractmethod
    def estimate_standard_errors_MLE(self):           # maybe not abstract method
        pass
    
    @abstractmethod
    def prepare_curvature_adjustment(self):
        # TODO: singular value decomp
        
        if not self.res.type=='MLE':
            raise TypeError('must optimize log-likelihood first')
            
        B = np.linalg.cholesky(self.MLEs_cov)

        L_inv = np.linalg.inv(np.linalg.cholesky(self.propcov))    # propcov only for MLE
        self.C = np.linalg.inv(B@L_inv)
        
        try:
            self.adj_propcov = np.linalg.inv(-hessian(self.adj_loglik)(self.res.x))
            
            # TODO: clean this up
            if not np.all(np.isfinite(self.adj_propcov)):       # use numerical diff
                hess = Hessian(self.adj_logpost)(self.res.x)
                self.adj_propcov = -np.linalg.inv(hess)
        except:
            hess = Hessian(self.adj_logpost)(self.res.x)
            self.adj_propcov = -np.linalg.inv(hess)
        return
    
    @abstractmethod
    def RW_MH(self, niter:int, adjusted:bool=False, acceptance_lag:int=1000, print_res:bool=True, **postargs):
        '''Random walk Metropolis-Hastings: samples the specified posterior'''
        
        # TODO: mcmc diagnostics
        
        if adjusted:
            posterior = self.adj_logpost
            propcov   = self.adj_propcov
            label = 'adjusted ' + self.label
            
        else:
            posterior = self.logpost
            propcov   = self.propcov
            label = self.label
    
        A     = np.zeros(niter, dtype=np.float64)
        U     = np.random.rand(niter)
            
        h = 2.38/np.sqrt(self.n_params)        
        props = h*np.random.multivariate_normal(np.zeros(self.n_params), propcov, size=niter)
        
        self.post_draws = np.zeros((niter, self.n_params))
        
        crnt_step = self.post_draws[0] = self.res.x
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
                
        return self.post_draws, A