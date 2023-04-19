from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from abc import ABC, abstractmethod

from time import time
from autograd import grad, hessian
from autograd.scipy import stats
from autograd.numpy import ndarray
from scipy.optimize import minimize, basinhopping

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.bayes.likelihoods_base import Likelihood

import autograd.scipy.linalg as spl
npl = np.linalg
fftn = np.fft.fftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


class DeWhittle(Likelihood):
    
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, use_taper:None|ndarray=None):
        super().__init__(z, grid, model, use_taper)
        
        
    def expected_periodogram(self, params: ndarray) -> ndarray:
        acf = self.cov_func(params, lags = None)
        return compute_ep(acf, self.grid.spatial_kernel, self.grid.mask) 

        
    def __call__(self, params: ndarray, I:None|ndarray=None, const:str='whittle') -> float: 
        params = np.exp(params)
        
        if I is None:
            I = self.I
            
        N = self.grid.n_points
        
        e_I = self.expected_periodogram(params)
        if const=='whittle':
            a=1/2
        else:
            a=1/N
            
        ll = -(a) * np.sum(np.log(e_I) + I / e_I)
        return ll
        
    def __repr__(self):
        return 'Debiased Whittle'
    
    def update_model_params(self, params: ndarray) -> None:
        return super().update_model_params(params)
    
    def cov_func(self, params: ndarray, lags: None|ndarray=None) -> ndarray:
        return super().cov_func(params, lags)
    
    def logprior(self, x: ndarray):
        return super().logprior(x)
        
    def logpost(self, x: ndarray):
        return super().logpost(x)
        
    def adj_loglik(self, x: ndarray):
        return super().adj_loglik(x)
        
    def adj_logpost(self, x: ndarray):
        return super().adj_logpost(x)
        
    def mymethod(self, x):
        super().mymethod(x)
    
    def fit(self, x0: None|ndarray, prior:bool = True, basin_hopping:bool = False, 
                                                       niter:int = 100, 
                                                       print_res:bool = True,
                                                       **optargs):
        
        return super().fit(x0=x0, prior=prior, basin_hopping=basin_hopping, niter=niter, print_res=print_res, **optargs)
    
    def sim_z(self, params: None|ndarray):
        return super().sim_z(params)
    
    def sim_MLEs(self, params: ndarray, niter:int=5000, t_random_field:bool=False, df:None|int=10, **optargs) -> ndarray:
        return super().sim_MLEs(params, niter, t_random_field, df, **optargs)
    
    
    def estimate_standard_errors_MLE(self, params: ndarray, monte_carlo:bool=False, niter:int=5000):           # maybe not abstract method
    
        if monte_carlo:
            return super().sim_MLEs(params, niter)
        else:
            pass
    
    def prepare_curvature_adjustment(self):           # maybe not abstract method
        return super().prepare_curvature_adjustment()
    
    def RW_MH(self, niter:int, adjusted:bool=False, acceptance_lag:int=1000, **postargs):
        return super().RW_MH(niter, adjusted=adjusted, acceptance_lag=acceptance_lag, **postargs)


class Whittle(Likelihood):
    
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, use_taper:None|ndarray=None):
        super().__init__(z, grid, model, use_taper)
        self.g = np.stack(np.meshgrid(*(np.arange(-n//2,n//2) for n in self.grid.n), indexing='ij'))  # for regular whittle


    def aliased_f(self, params: ndarray) -> ndarray:
        '''
        Computes the aliased spectral density in O(|n|log|n|) time for the given covariance model
        For small grids may need to upsample
        '''
        # TODO: ask Arthur why ifftshift
        
        acf = ifftshift(self.cov_func(params, self.g))  # ifftshift again?
        f = np.real(fftn(fftshift(acf)))
        assert np.all(f>0)
        return f

    def __call__(self, params: ndarray, I:None|ndarray=None) -> float:
        '''Computes 2d Whittle likelihood in O(|n|log|n|) time'''
        params = np.exp(params)
        
        if I is None:
            I = self.I
            
        f = self.aliased_f(params)           # this may be unstable for small grids/nugget
        
        ll = -(1/2) * np.sum(np.log(f) + I / f)
        return ll
        
    def __repr__(self):
        return 'Whittle'
    
    def update_model_params(self, params: ndarray) -> None:
        return super().update_model_params(params)
    
    def cov_func(self, params: ndarray, lags: None|ndarray=None) -> ndarray:
        return super().cov_func(params, lags)
    
    def logprior(self, x: ndarray):
        return super().logprior(x)
        
    def logpost(self, x: ndarray):
        return super().logpost(x)
        
    def adj_loglik(self, x: ndarray):
        return super().adj_loglik(x)
        
    def adj_logpost(self, x: ndarray):
        return super().adj_logpost(x)
        
    def mymethod(self, x):
        super().mymethod(x)
    
    def fit(self, x0: None|ndarray, prior:bool = True, basin_hopping:bool = False, 
                                                       niter:int = 100, 
                                                       print_res:bool = True,
                                                       **optargs):
        
        return super().fit(x0=x0, prior=prior, basin_hopping=basin_hopping, niter=niter, print_res=print_res, **optargs)
    
    def sim_z(self, params: None|ndarray):
        return super().sim_z(params)
    
    def sim_MLEs(self, params: ndarray, niter:int=5000, const:str='whittle', **optargs) -> ndarray:
        return super().sim_MLEs(self, params, niter, const, **optargs)
    
    
    def estimate_standard_errors_MLE(self, params: ndarray, monte_carlo:bool=False, niter:int=5000):           # maybe not abstract method
    
        if monte_carlo:
            return super().sim_MLEs(params, niter)
        else:
            pass   # Theorem 3 from paper
    
    def prepare_curvature_adjustment(self):           # maybe not abstract method
        return super().prepare_curvature_adjustment()
    
    def RW_MH(self, niter:int, adjusted:bool=False, acceptance_lag:int=1000, **postargs):
        return super().RW_MH(niter, adjusted=adjusted, acceptance_lag=acceptance_lag, **postargs)


class Gaussian(Likelihood):


    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel):
        
        if grid.n_points>10000:
            ValueError('Too many observations for Gaussian likelihood')
            
        self.norm2 = np.sum((lags**2 for lags in grid.lag_matrix))
        super().__init__(z, grid, model, use_taper=None)

    def __call__(self, params: ndarray, z:None|ndarray=None) -> float:
        '''Computes Gaussian likelihood in O(|n|^3) time'''
        params = np.exp(params)
        
        if z is None:
            z = self.z
        
        N = self.n_points
        covMat = self.cov_func(params, lags=self.norm2)
   
        L  = npl.cholesky(covMat)
        S1 = spl.solve_triangular(L,   z.flatten(),  lower=True)
        S2 = spl.solve_triangular(L.T, S1, lower=False)
       
        ll = -np.sum(np.log(np.diag(L)))             \
              -0.5*np.dot(z.flatten(),S2)        \
              -0.5*N*np.log(2*np.pi)
        return ll
        
    def __repr__(self):
        return 'Gaussian'
    
    def update_model_params(self, params: ndarray) -> None:
        return super().update_model_params(params)
    
    def cov_func(self, params: ndarray, lags: None|ndarray=None) -> ndarray:
        
        if lags is None:
            lags = self.norm2

        self.update_model_params(params)
        return self.model(lags, time_domain=True)

    
    def logprior(self, x: ndarray):
        return super().logprior(x)
        
    def logpost(self, x: ndarray):
        return super().logpost(x)
        
    def adj_loglik(self, x: ndarray):
        raise ValueError('too ')
        return super().adj_loglik(x)
        
    def adj_logpost(self, x: ndarray):
        return super().adj_logpost(x)
        
    def mymethod(self, x):
        super().mymethod(x)
    
    def sim_z(self, params: None|ndarray):
        return super().sim_z(params)
    
    def fit(self, x0: None|ndarray, prior:bool = True, basin_hopping:bool = False, 
                                                        niter:int = 100, 
                                                        print_res:bool = True,
                                                        **optargs):
        
        return super().fit(x0=x0, prior=prior, basin_hopping=basin_hopping, niter=niter, print_res=print_res, **optargs)
    
    def sim_MLEs(self, params: ndarray, niter:int=5000, const:str='whittle', **optargs) -> ndarray:
        return super().sim_MLEs(self, params, niter, const, **optargs)
    
    
    def estimate_standard_errors_MLE(self, params: ndarray, monte_carlo:bool=False, niter:int=5000):           # maybe not abstract method
    
        if monte_carlo:
            return super().sim_MLEs(params, niter)
        else:
            pass   # https://en.wikipedia.org/wiki/Fisher_information#Multivariate_normal_distribution
    
    def prepare_curvature_adjustment(self):           # maybe not abstract method
        return super().prepare_curvature_adjustment()
    
    def RW_MH(self, niter:int, adjusted:bool=False, acceptance_lag:int=1000, **postargs):
        return super().RW_MH(niter, adjusted=adjusted, acceptance_lag=acceptance_lag, **postargs)

