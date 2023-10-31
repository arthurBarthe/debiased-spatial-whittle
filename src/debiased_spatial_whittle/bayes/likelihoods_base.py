from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from abc import ABC, abstractmethod

from autograd import grad, hessian
from numdifftools import Hessian
from autograd.scipy import stats
from autograd.numpy import ndarray
from scipy.optimize import minimize, basinhopping

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, TSamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.bayes.funcs import transform, compute_gradient, compute_hessian, svd_decomp
from typing import Union, Optional, Callable, List

fftn = np.fft.fftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

inv = np.linalg.inv
cholesky = np.linalg.cholesky

class Likelihood(ABC):

    def __init__(self, z: ndarray, grid: RectangularGrid,
                 model: CovarianceModel, nugget: Optional[float] = 0.1,
                 use_taper: Union[None, ndarray]=None, transform_func: Optional[Callable] = None):
        
        self._z = z
        self.grid = grid
        # TODO: make this property
        self.n = self.grid.n
        
        self.use_taper = use_taper
        self.periodogram = Periodogram(taper=use_taper)
        self._I = self.periodogram(z)   # TODO: make periodogram cached_property
        self._constant = 1/2
        
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
        
        if transform_func is None:             # TODO: include in Parameter class
            self.transform = self._transform
            self._transform_flag = False
        else:
            self.transform = transform_func
            self._transform_flag = True
        
            
            
    @property
    def constant(self):
        '''multiplicative constant on the log-likelihood, only for Whittle and DeWhittle'''
        return self._constant
    
    @constant.setter
    def constant(self, c: float):
        self._constant = c
    
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
    
    def _transform(self, x: ndarray, inv:bool=True) -> ndarray:
        return x
    
    @property
    def transform_flag(self):
        return self._transform_flag
    
    @abstractmethod
    def __call__(self, x: ndarray) -> float: # loglik
        pass
    
    @property
    def label(self) -> str:
        return self.__class__.__name__
    
    def __repr__(self):
        return f'{self.label} likelihood with {self.model.name} and n={self.n}'
    
    def update_model_params(self, params: ndarray) -> None:
       all_params = self.model.params
       updates = dict(zip(all_params.names, params))
       all_params.update_values(updates)
       return
   
    def cov_func(self, params: ndarray, lags: Union[None,List[ndarray]] =None, **cov_args) -> List[ndarray]:
        '''compute covariance func on a grid of lags given parameters'''
        
        # TODO: only for dewhittle and whittle
        if lags is None:
            lags = self.grid.lags_unique

        self.update_model_params(params)
        return self.model(np.stack(lags), **cov_args)
    
    def fit(self,  
            x0: Union[None, ndarray]=None,
            basin_hopping:bool = False,
            approx_grad:bool=False,
            print_res:bool = True,
            save_res:bool=True,
            loglik_kwargs: Union[None,Callable]=None,
            **opt_kwargs):

        '''
        A general optimizer of the log-likelihood. Includes optional
        global optimizer.
        '''
        # TODO: test this!
        
        attribute = 'MLE'
        
        if x0 is None:
            x0 = self.transform(np.ones(self.n_params), inv=False)        
            
        if not self.transform_flag:
            bounds = self.model.param_bounds[:len(x0)]
        else:
            bounds = None
        
        if loglik_kwargs is None:
            loglik_kwargs = dict()
            
        def obj(x):     return -1 / (self.n_points * self.constant) * self(x, **loglik_kwargs)  # minimize rescaled negative loglik
            
        gradient = False if approx_grad else grad(obj)
        
        if basin_hopping:          # for global optimization
            minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient, 'bounds': bounds}
            res = basinhopping(obj, x0, minimizer_kwargs=minimizer_kwargs, **opt_kwargs)   # niter!!
            success = res.lowest_optimization_result['success']
        else:            
            res = minimize(x0=x0, 
                           fun=obj,
                           jac=gradient,
                           method='L-BFGS-B',
                           bounds=bounds,
                           **opt_kwargs)
            
            success = res['success']
            
        if not success or obj(res.x)>obj(x0):
            print('Optimizer failed!')
            # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
            
        res['type'] = attribute
        res['BIC']  = len(x0) * np.log(self.n_points) - 2*self(res.x)
        if print_res:
            print(f'{self.label} {attribute}:  {self.transform(res.x).round(3)}')
            
        if save_res:
            setattr(self, 'res', res)
            
        return res
    
    def adj_loglik(self, x: ndarray, C, **loglikargs) -> float: 
        return self(self.res.x + C @ (x - self.res.x), **loglikargs)

    
    def sim_z(self, params: Union[None, ndarray]=None):
        '''simulated z (random field) at params'''
        self.update_model_params(params)
        sampler = self.sampler_model(self.model, self.grid)
        # print(sampler.gaussian_sampler.sampling_grid.n)
        return sampler()
    
    def sim_MLEs(self, params: ndarray, niter:int=5000, print_res:bool=True, approx_grad:bool = False, **opt_kwargs) -> ndarray:
        '''simulation approximation of the sampling distribution of MLE at params'''
        i = 0
        self.MLEs = np.zeros((niter, self.n_params), dtype=np.float64)
        while niter>i:
            
            z = self.sim_z(params)

            if False:
                # print(z[4,11])
                import matplotlib.pyplot as plt
                plt.imshow(z, origin='lower')
                plt.show()
                
            loglik_kwargs = {'z':z}
            res = self.fit(x0=params,
                           print_res=False, 
                           save_res=False, 
                           loglik_kwargs=loglik_kwargs, 
                           **opt_kwargs)
            
            
            if res.x[0] > 100:
                print(res['success'])
                import matplotlib.pyplot as plt
                plt.imshow(z)
                plt.show()
                
            if not res['success'] or np.linalg.norm(res.x - params)>1000:
                print('failed out fit method')
                continue

            if print_res:
                print(f'{i+1})   MLE:  {self.transform(res.x).round(3)}')
            self.MLEs[i] = res.x
            i+=1
            
        self.MLEs_cov = np.cov(self.MLEs.T)
        return self.MLEs
    
    
    def sim_J_matrix(self, params: ndarray,
                           niter:int=5000, 
                           print_res:bool=True, 
                           approx_grad:bool = False,
                           **opt_kwargs) -> ndarray:
        '''
        Estimate via simulation J(x) matrix, var[ grad(ll) ].
        '''
        # TODO: more testing!
        # TODO: Jhat now on 1/n scale!
        
        def obj(x, **loglik_kwargs):
            '''rescaled log-lik'''
            return -1 / (self.n_points * self.constant) * self(x, **loglik_kwargs) 
    
        i = 0
        self.grad_at_params = np.zeros((niter, self.n_params), dtype=np.float64)
        while niter>i:
            
            z = self.sim_z(params)
            loglik_kwargs = {'z':z}
            
            # TODO: func kwargs?
            grad_at_params = compute_gradient(obj, params, approx_grad=approx_grad, **loglik_kwargs)   
           
            if not np.all(np.isfinite(grad_at_params)):
                continue
            
            if print_res:
                print(f'\riteration {i+1})', end='')
            self.grad_at_params[i] = grad_at_params
            i+=1
        
        self.Jhat = np.cov( self.grad_at_params.T )      # multiply by (2/N) for original dewhittle constants
        return self.grad_at_params
        
    
    def sim_H_and_J_matrices(self, params: ndarray,
                             niter:int=5000, 
                             print_res:bool=True, 
                             approx_grad:bool = True,
                             **opt_kwargs) -> ndarray:
        '''
        Estimate via simulation H(x) and J(x) matrices. For models which the 
        fisher (H) cannot be computed analyitcally (e.g. Matern).
        '''
        # TODO: more testing!
        def obj(x, **loglik_kwargs):
            '''rescaled log-lik'''
            return -1 / (self.n_points * self.constant) * self(x, **loglik_kwargs) 
    
    
        i = 0
        self.grad_at_params = np.zeros((niter, self.n_params), dtype=np.float64)
        self.hess_at_params = np.zeros((niter, self.n_params, self.n_params), dtype=np.float64)
        while niter>i:
            
            z = self.sim_z(params)
            loglik_kwargs = {'z':z}
            
            grad_at_params = compute_gradient(obj, params, approx_grad=approx_grad, **loglik_kwargs)   # TODO: func kwargs?
            hess_at_params = compute_hessian(obj, params, approx_grad=approx_grad, **loglik_kwargs)
           
            if not np.all(np.isfinite(grad_at_params)) or np.all(np.isfinite(hess_at_params)):
                continue
            
            if print_res:
                print(f'\riteration {i+1})', end='') # grad: {grad_at_params.round(3)}')
            self.grad_at_params[i] = grad_at_params
            self.hess_at_params[i] = hess_at_params
            i+=1
        
        self.Hhat = - np.mean(self.hess_at_params,axis=0)         # multiply by (2/N) for original dewhittle constants
        self.Jhat = np.cov( self.grad_at_params.T )
        return self.grad_at_params, self.hess_at_params
    

    # @abstractmethod
    # def estimate_standard_errors_MLE(self):           # maybe not abstract method
    #     pass

    # compute_C is when we have analytic (approx) J matrix
    
    # TODO: USE CHOLESKY INSTEAD OF SVD_DECOMP, BETER RESULTS WITH CHOLESKY

    def compute_C2(self):
        '''Compute C matrix for posterior adjustment with MC estimate of J.'''
        if hasattr(self, 'Hhat'):
            H_ = self.Hhat
        elif hasattr(self, 'H'):
            H_ = self.H
        
        # TODO: analytic J?
        if hasattr(self, 'Jhat'):
            Jhat = self.Jhat
        
        M_A = cholesky( H_ @ inv(Jhat) @ H_ )
        M   = cholesky( H_ )
        self.C2 = inv(M) @ M_A / np.sqrt(self.n_points / 2)   # re-adjusting to 1/2 scale!
        return
        
    def compute_C3(self, mle: ndarray):
        if hasattr(self, 'Hhat'):
            H_ = self.Hhat
        elif hasattr(self, 'H'):
            H_ = self.H
        
        # TODO: analytic J?
        if hasattr(self, 'Jhat'):
            Jhat = self.Jhat
        
        M_A = cholesky( H_ @ inv(Jhat) @ H_ )
        M   = cholesky(compute_hessian(self, mle))    # observed fisher
        self.C3 = inv(M) @ M_A
        return
    
    def compute_C4(self, mle: ndarray):
        '''
        C = M^-1 M_A,
        M_A = svd(H J^-1 H),
        M   = svd(H).
        '''
        M_A = cholesky(inv(self.MLEs_cov))
        
        M = cholesky(compute_hessian(self, mle))  # this is the observed fisher
        self.C4 = inv(M) @ M_A
        return
    
    def compute_C5(self, mle: ndarray):
        '''First  way of trying the adjustment.'''
        # TODO: change in likelihoods.py!!

        B = cholesky(self.MLEs_cov)     # this is inv(H) J inv(H)
        
        # TODO: only autograd propcov
        propcov = compute_hessian(self, mle, inv=True)     # this is inv(H), inv of observed fisher
        L_inv = inv(cholesky(propcov))    # propcov only for MLE
        self.C5 = inv(B @ L_inv)
        return
    
    def compute_C5_2(self, mle: ndarray):
        B = svd_decomp(self.MLEs_cov)
        
        propcov = compute_hessian(self, mle, inv=True)
        L_inv = inv(svd_decomp(propcov))    # propcov only for MLE
        self.C5_2 = inv(B @ L_inv)
        return
    
    def compute_C6(self, mle: ndarray):
        # TODO: change in likelihoods.py!!
        H = self.H
        B = cholesky(inv(H) @ self.Jhat @ inv(H))     # this is inv(H) J inv(H)
        
        # TODO: only autograd propcov
        propcov = compute_hessian(self, mle, inv=True)     # this is inv(H), inv of observed fisher
        L_inv = inv(cholesky(propcov))    # propcov only for MLE
        self.C6 = inv(B @ L_inv)
        return
    
    
