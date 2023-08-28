from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from autograd.numpy import ndarray

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.bayes.likelihoods_base import Likelihood
from debiased_spatial_whittle.bayes.funcs import transform

from typing import Union, Optional, Dict, Callable
from functools import cached_property
import autograd.scipy.linalg as spl
npl = np.linalg
fftn = np.fft.fftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
fftfreq = np.fft.fftfreq


class DeWhittle(Likelihood):
    
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, nugget: float, use_taper: Union[None, ndarray]=None, transform_func: Optional[Callable] = None):
        super().__init__(z, grid, model, nugget, use_taper, transform_func)
        self.frequency_mask = None
    
    @property
    def frequency_mask(self):
        if self._frequency_mask is None:
            return 1
        else:
            return self._frequency_mask

    @frequency_mask.setter
    def frequency_mask(self, value: np.ndarray):
        """
        Define a mask in the spectral domain to fit only certain frequencies

        Parameters
        ----------
        value
            mask of zeros and ones
        """
        if value is not None:
            assert value.shape == self.grid.n, "shape mismatch between mask and grid"
        self._frequency_mask = value
    
    def expected_periodogram(self, params: ndarray, **cov_args) -> ndarray:
        acf = self.cov_func(params, lags = None, **cov_args)
        return compute_ep(acf, self.grid.spatial_kernel, self.grid.mask)

        
    def __call__(self, params: ndarray, z:Optional[ndarray]=None, **cov_args) -> float: 
        
        params = self.transform(params, inv=True)
        
        if z is None:
            I = self.I
        else:
            I = self.periodogram(z)
                    
        e_I = self.expected_periodogram(params, **cov_args)
        
        # TODO: constants?            
        ll = -(1/2) * np.sum( (np.log(e_I) + I / e_I) * self.frequency_mask )
        return ll
    

class Whittle(Likelihood):
    
    def __init__(self, z: ndarray, grid: RectangularGrid,
                 model: CovarianceModel,
                 nugget: float, use_taper:Optional[ndarray]=None,
                 aliased_grid_shape:tuple = (3,3),
                 transform_func: Optional[Callable] = None):
        
        
        super().__init__(z, grid, model, nugget, use_taper, transform_func)
        
        self._aliased_grid_shape = aliased_grid_shape   # aliasing
        self.frequency_mask = None
    
    @cached_property
    def freq_grid(self):
        '''grid of frequencies to compute covariance function''' 
        return np.meshgrid(*(2*np.pi*fftfreq(n) for n in self.n), indexing='ij')         # / (delta*n)?
    
    @property
    def aliased_grid_shape(self):
        '''size of the infinite grid for aliaising in spectral density'''
        return self._aliased_grid_shape
    
    @cached_property
    def aliased_grid(self):
        '''grid to compute the aliased spectral density, truncated infinite sum grid'''
        shape = self.aliased_grid_shape
        delta = self.grid.delta
        return np.meshgrid(*(2*np.pi*np.arange(-(n//2), n//2+1)/delta[i] for i,n in enumerate(shape)), indexing='ij')    # np.arange(0,1) for non-aliased version
    

    @property
    def frequency_mask(self):
        if self._frequency_mask is None:
            return 1
        else:
            return self._frequency_mask

    @frequency_mask.setter
    def frequency_mask(self, value: np.ndarray):
        """
        Define a mask in the spectral domain to fit only certain frequencies

        Parameters
        ----------
        value
            mask of zeros and ones
        """
        if value is not None:
            assert value.shape == self.grid.n, "shape mismatch between mask and grid"
        self._frequency_mask = value

    
    def f(self, params: ndarray) -> ndarray:
        '''
        Computes the aliased spectral density for given model
        '''
        
        self.update_model_params(params)        
        f = self.model.f(self.freq_grid, self.aliased_grid)
        return f

    @cached_property
    def freq_grid_for_fft(self):    
        '''grid for fft of covariance func for regular whittle'''
        return np.stack(np.meshgrid(*(np.arange(-n//2,n//2) for n in self.n), indexing='ij'))
    
    def aliased_f_fft(self, params: ndarray, **cov_args) -> ndarray:
        '''
        Computes the aliased spectral density in O(|n|log|n|) time for the given covariance model
        For small grids may need to upsample
        '''
        
        acf = ifftshift(self.cov_func(params, self.freq_grid_for_fft, **cov_args))  # undoing ifftshift
        f = np.real(fftn(fftshift(acf)))
        assert np.all(f>0)
        return f

    def __call__(self, params: ndarray, z:Optional[ndarray]=None, **kwargs) -> float:
        '''Computes 2d Whittle likelihood'''
        # TODO: add spectral density
        # TODO: include optional arguemtn to input infnite sum grid
        params = self.transform(params, inv=True)
        
        if z is None:
            I = self.I
        else:
            I = self.periodogram(z)
            
        f = self.f(params)           # this may be unstable for small grids/nugget
        
        ll = -(1/2) * np.sum(  (np.log(f) + I / f) * self.frequency_mask  )
        return ll
    

class Gaussian(Likelihood):

    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, nugget: float, transform_func: Optional[Callable] = None):
        
        if grid.n_points>10000:
            ValueError('Too many observations for Gaussian likelihood')
        
        super().__init__(z, grid, model, nugget=nugget, use_taper=None, transform_func=transform_func)

    def __call__(self, params: ndarray, z:Optional[ndarray]=None) -> float:
        '''Computes Gaussian likelihood in O(|n|^3) time'''
        params = self.transform(params, inv=True)
        
        if z is None:
            z = self.z
        
        N = self.n_points
        covMat = self.cov_func(params, lags=self.grid.lag_matrix)
   
        L  = npl.cholesky(covMat)
        S1 = spl.solve_triangular(L,   z.flatten(),  lower=True)
        S2 = spl.solve_triangular(L.T, S1, lower=False)
       
        ll = -np.sum(np.log(np.diag(L)))         \
              -0.5* np.dot(z.flatten(),S2)        \
              -0.5*N*np.log(2*np.pi)
        return ll
        
    def adj_loglik(self, x: ndarray):
        # TODO: better error message?
        raise ValueError('too costly')
        
