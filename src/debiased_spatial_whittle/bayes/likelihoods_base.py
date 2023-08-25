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
from debiased_spatial_whittle.bayes.funcs import transform, compute_propcov
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
    def adj_loglik(self, x: ndarray, **loglikargs) -> float: 
        return self(self.res.x + self.C @ (x - self.res.x), **loglikargs)

    
    @abstractmethod
    def sim_z(self, params: Union[None, ndarray]=None):
            
        self.update_model_params(params)
        sampler = self.sampler_model(self.model, self.grid)
        # print(sampler.gaussian_sampler.sampling_grid.n)
        return sampler()
    
    @abstractmethod
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
            
            def obj(x):     return -self(x, **loglik_kwargs)
            
            if not approx_grad:
                gradient = grad(obj)
            else:
                gradient = False
                
                
            x0 = self.transform(params, inv=False)
                
            # TODO: no basin-hopping
            res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **opt_kwargs)
            if res.x[0] > 100:
                print(res['success'])
                import matplotlib.pyplot as plt
                plt.imshow(z)
                plt.show()
            if not res['success'] or np.linalg.norm(res.x - params)>1000:
                continue

            else:
                if print_res:
                    print(f'{i+1})   MLE:  {self.transform(res.x).round(3)}')
                self.MLEs[i] = res.x
                i+=1
            
        self.MLEs_cov = np.cov(self.MLEs.T)
        # self.prepare_curvature_adjustment()      # compute C matrix for posterior adjustment
        return self.MLEs


    @abstractmethod
    def estimate_standard_errors_MLE(self):           # maybe not abstract method
        pass
    
    @abstractmethod
    def prepare_curvature_adjustment(self, mle: ndarray):
        # TODO: singular value decomp
        # TODO: change in likelihoods.py!!

        B = cholesky(self.MLEs_cov)
        
        # TODO: only autograd propcov
        propcov = compute_propcov(self, mle)
        L_inv = inv(cholesky(propcov))    # propcov only for MLE
        self.C = inv(B @ L_inv)
        return
    
    
    # @abstractmethod
    # def sim_var_grad(self, params: ndarray, niter:int=5000, print_res:bool=True, approx_grad:bool = False, **opt_kwargs) -> ndarray:
    #     '''estimate J(x) matrix, var[grad(ll)]'''
    #     # TODO: more testing!
        
    #     if self.transform_params:
    #         x = transform(params, inv=False)
            
    #     i = 0
    #     self.grad_at_params = np.zeros((niter, self.n_params), dtype=np.float64)
    #     self.hess_at_params = np.zeros((niter, self.n_params, self.n_params), dtype=np.float64)
    #     while niter>i:
            
    #         z = self.sim_z(params)
    #         loglik_kwargs = {'z':z}
            
    #         grad_at_params = grad(self)(x, **loglik_kwargs)
    #         hess_at_params = hessian(self)(x, **loglik_kwargs)
           
    #         if not np.all(np.isfinite(grad_at_params)):    # TODO: approx grad!!
    #             continue
            
    #         if not np.all(np.isfinite(hess_at_params)):    # TODO: approx grad!!
    #             continue

    #         else:   # TODO: fix this for if one false, one true
    #             if print_res:
    #                 print(f'{i+1})') # grad: {grad_at_params.round(3)}')
    #             self.grad_at_params[i] = grad_at_params
    #             self.hess_at_params[i] = hess_at_params
    #             i+=1
            
    #     self.J = np.cov(self.grad_at_params.T)
    #     # self.prepare_curvature_adjustment()      # compute C matrix for posterior adjustment
    #     return self.J