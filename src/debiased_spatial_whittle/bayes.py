# import autograd.numpy as np
from .backend import BackendManager
# TODO: why .backend??
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from autograd import grad, hessian
from autograd.scipy import stats
from autograd.numpy import ndarray
from scipy.optimize import minimize, basinhopping
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from .models import CovarianceModel
# TODO: fix backend imports for all imported modules!!

ifftshift = np.fft.ifftshift

class DeWhittle:
    # TODO: include time domain and regular whittle likelihoods
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, use_taper:None|ndarray=None):
        self._z = z
        self.grid = grid
        
        self.use_taper = use_taper
        self.periodogram = Periodogram(taper=use_taper)
        self._I = self.periodogram(z)
        
        self.model = model
        self.free_params = model.free_params
        self.n_params = len(self.free_params)
        
    @property
    def z(self):
        return self._z
    
    @property
    def I(self):
        return self._I
        
    def cov_func(self, params: ndarray, lags:None|list[ndarray, ...] = None) -> list[ndarray, ...]:
        '''compute covariance func on a grid of lags given parameters'''
        # TODO: parameter transform is temporary
        params = np.exp(params)
        if lags is None:
            lags = self.grid.lags_unique
            
        free_params = self.model.params        
        updates = dict(zip(free_params.names, params))
        free_params.update_values(updates)
        return ifftshift(self.model(self.grid.lags_unique))
        
        
    def expected_periodogram(self, params: ndarray) -> ndarray:
        acf = self.cov_func(params, lags = None)
        return compute_ep(acf, self.grid.spatial_kernel, self.grid.mask) 
    
    
    def loglik(self, params: ndarray) -> float:
        # TODO: transform params
        N = self.grid.n_points
        
        e_I = self.expected_periodogram(params)
        # TODO: may need to change constant 1/N    
        ll = -(1/N) * np.sum(np.log(e_I) + self.I / e_I)
        return ll
    
    def logprior(self, x: ndarray) -> float:
        '''uninformative prior on the transformed (unrestricted space)'''
        k = self.n_params
        return stats.multivariate_normal.logpdf(x, np.zeros(k), cov=np.eye(k)*100)
    
    
    def logpost(self, x: ndarray) -> float:
        return self.loglik(x) + self.logprior(x)
    
    def fit(self, x0: None|ndarray, prior:bool = True,  basin_hopping:bool = False, 
                                                        niter:int = 100, 
                                                        label: str = 'debiased Whittle', 
                                                        print_res:bool = True, **optargs):
        '''fit the model to data - includes optional global optimizer'''
        
        if x0 is None:
            x0 = np.zeros(self.n_params)
            
        if prior:
            label += ' MAP'
            def obj(x):     return -self.logpost(x)
        else:
            label += ' MLE'
            def obj(x):     return -self.loglik(x)
            
        if basin_hopping:          # for global optimization
            minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': grad(obj)}
            self.res = basinhopping(obj, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, **optargs) # seed=234230
            success = self.res.lowest_optimization_result['success']
        else:            
            self.res = minimize(x0=x0, fun=obj, jac=grad(obj), method='L-BFGS-B', **optargs)
            success = self.res['success']

        self.BIC = self.n_params*np.log(self.grid.n_points) - 2*self.loglik(self.res.x)         # negative logpost
            
        if not success:
            print('Optimizer failed!')
            # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
        
        if print_res:
            print(f'{label}:  {np.round(np.exp(self.res.x),3)}')
        
        try:
            self.propcov = np.linalg.inv(-hessian(self.logpost)(self.res.x))
        except np.linalg.LinAlgError:
            print('Singular propcov')
            self.propcov = False
            
        return self.res, self.propcov