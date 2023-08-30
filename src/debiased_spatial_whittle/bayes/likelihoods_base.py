from typing import Optional, Callable

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from abc import ABC, abstractmethod

from autograd import grad, hessian
from numdifftools import Hessian
from autograd.scipy import stats
from autograd.numpy import ndarray
from scipy.optimize import minimize

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, TSamplerOnRectangularGrid, SquaredSamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.bayes.funcs import transform, compute_gradient, compute_hessian, svd_decomp, fit
from typing import Union

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
        
        if transform_func is None:
            self.transform = self._transform
        else:
            self.transform = transform_func
    
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
   
    def cov_func(self, params: ndarray, lags: Optional[list[ndarray, ...]] =None, **cov_args) -> list[ndarray, ...]:
        '''compute covariance func on a grid of lags given parameters'''
        
        # TODO: only for dewhittle and whittle
        if lags is None:
            lags = self.grid.lags_unique

        self.update_model_params(params)
        return self.model(np.stack(lags), **cov_args)
    
    def fit(self, **fit_kwargs):
        # TODO: copy docstring and kwargs over from fit
        return fit(self, **fit_kwargs)
    
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
                           included_prior=False, 
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
    
        i = 0
        self.grad_at_params = np.zeros((niter, self.n_params), dtype=np.float64)
        while niter>i:
            
            z = self.sim_z(params)
            loglik_kwargs = {'z':z}
            
            # TODO: func kwargs?
            grad_at_params = compute_gradient(self, params, approx_grad=approx_grad, **loglik_kwargs)   
           
            if not np.all(np.isfinite(grad_at_params)):
                continue
            
            if print_res:
                print(f'\riteration {i+1})', end='')
            self.grad_at_params[i] = grad_at_params
            i+=1
        
        self.Jhat = np.cov( (2 / self.n_points) * self.grad_at_params.T)      # multiply by (2/N) for original dewhittle constants
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
    
        i = 0
        self.grad_at_params = np.zeros((niter, self.n_params), dtype=np.float64)
        self.hess_at_params = np.zeros((niter, self.n_params, self.n_params), dtype=np.float64)
        while niter>i:
            
            z = self.sim_z(params)
            loglik_kwargs = {'z':z}
            
            grad_at_params = compute_gradient(self, params, approx_grad=approx_grad, **loglik_kwargs)   # TODO: func kwargs?
            hess_at_params = compute_hessian(self, params, approx_grad=approx_grad, **loglik_kwargs)
           
            if not np.all(np.isfinite(grad_at_params)) or np.all(np.isfinite(hess_at_params)):
                continue
            
            if print_res:
                print(f'\riteration {i+1})', end='') # grad: {grad_at_params.round(3)}')
            self.grad_at_params[i] = grad_at_params
            self.hess_at_params[i] = hess_at_params
            i+=1
        
        self.Hhat = - 2 * np.mean(self.hess_at_params,axis=0)/ self.n_points    # multiply by (2/N) for original dewhittle constants
        self.Jhat = np.cov( (2 / self.n_points) * self.grad_at_params.T)
        return self.grad_at_params, self.hess_at_params
    

    # @abstractmethod
    # def estimate_standard_errors_MLE(self):           # maybe not abstract method
    #     pass

    def compute_C(self):
        '''compute C matrix for posterior adjustment'''
        if hasattr(self, 'Hhat'):
            H_ = self.Hhat
        elif hasattr(self, 'H'):
            H_ = self.H
        
        # TODO: analytic J?
        if hasattr(self, 'Jhat'):
            Jhat = self.Jhat
        
        M_A = svd_decomp( H_ @ inv(Jhat) @ H_ )
        M = svd_decomp( (self.n_points / 2) * H_ )   # TODO: THE CONSTANTS!!
        self.C = inv(M) @ M_A
        return
        
    
    def compute_C_old(self, mle: ndarray):
        # TODO: change in likelihoods.py!!

        B = cholesky(self.MLEs_cov)     # this is inv(H) J inv(H)
        
        # TODO: only autograd propcov
        propcov = compute_hessian(self, mle, inv=True)     # this is inv(H), inv of observed fisher
        L_inv = inv(cholesky(propcov))    # propcov only for MLE
        self.C1 = inv(B @ L_inv)
        return
    
    def compute_C2(self, mle: ndarray):
        B = svd_decomp(self.MLEs_cov)
        
        propcov = compute_hessian(self, mle, inv=True)
        L_inv = inv(svd_decomp(propcov))    # propcov only for MLE
        self.C2 = inv(B @ L_inv)
        return
    
    def compute_C3(self, mle: ndarray):
        '''
        C = M^-1 M_A
        M_A = svd(H J^-1 H)
        M   = svd(H)
        '''
        M_A = svd_decomp(inv(self.MLEs_cov))
        
        M = svd_decomp(compute_hessian(self, mle))  # this is the observed fisher
        self.C3 = inv(M) @ M_A
        return
    
    def compute_C4(self, mle: ndarray):
        '''
        C = M^-1 M_A
        M_A = svd(H J^-1 H)
        M   = svd(H)
        '''
        # TODO: Use theortical matrices!
        # M_A = svd_decomp(inv(self.MLEs_cov))
        
        # M = svd_decomp(compute_hessian(self, mle))  # this is the observed fisher
        # self.C3 = inv(M) @ M_A
        return
