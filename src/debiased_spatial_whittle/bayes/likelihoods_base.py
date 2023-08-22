from typing import Optional

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
from debiased_spatial_whittle.bayes.funcs import transform
from typing import Union

fftn = np.fft.fftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


class Likelihood(ABC):
    
    def __init__(self, z: ndarray, grid: RectangularGrid,
                 model: CovarianceModel, nugget: Optional[float] = 0.1,
                 use_taper: Union[None, ndarray]=None, transform_params: bool=True):
        
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
        
        self._transform_params = transform_params
        
    
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
    
    @property
    def transform_params(self):
        return self._transform_params
    
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
    
    
    # @abstractmethod
    # def logprior(self, x: ndarray) -> float:
    #     '''uninformative prior on the transformed (unrestricted space)'''
    #     k = self.n_params
    #     return stats.multivariate_normal.logpdf(x, np.zeros(k), cov=np.eye(k)*100)


    # @abstractmethod
    # def logpost(self, x: ndarray, **loglik_kwargs) -> float:
    #     return self(x, **loglik_kwargs) + self.logprior(x)

    
    @abstractmethod
    def adj_loglik(self, x: ndarray, **loglikargs) -> float: 
        return self(self.res.x + self.C @ (x - self.res.x), **loglikargs)

    # @abstractmethod
    # def adj_logpost(self, x: ndarray) -> float:
    #     return self.adj_loglik(x) + self.logprior(x)    
    
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
            
            _z = self.sim_z(params)

            if False:
                # print(_z[4,11])
                import matplotlib.pyplot as plt
                plt.imshow(_z, origin='lower')
                plt.show()
                
            loglik_kwargs = {'z':_z}
            
            def obj(x):     return -self(x, **loglik_kwargs)
            
            if not approx_grad:
                gradient = grad(obj)
            else:
                gradient = False
                
            if self.transform_params:
                x0 = transform(params, inv=False)
            else:
                x0 = params
                
            # TODO: no basin-hopping
            res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **opt_kwargs)

            if not res['success']:
                continue

            else:
                if print_res:
                    print(f'{i+1})   MLE:  {np.round(np.exp(res.x),3)}')
                self.MLEs[i] = res.x
                i+=1
            
        self.MLEs_cov = np.cov(self.MLEs.T)
        self.prepare_curvature_adjustment()      # compute C matrix for posterior adjustment
        return self.MLEs


    @abstractmethod
    def estimate_standard_errors_MLE(self):           # maybe not abstract method
        pass
    
    @abstractmethod
    def prepare_curvature_adjustment(self):
        # TODO: singular value decomp

        B = np.linalg.cholesky(self.MLEs_cov)

        L_inv = np.linalg.inv(np.linalg.cholesky(self.propcov))    # propcov only for MLE
        self.C = np.linalg.inv(B@L_inv)
        return