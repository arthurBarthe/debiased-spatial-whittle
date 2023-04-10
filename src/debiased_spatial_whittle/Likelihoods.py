from .backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from abc import ABC, abstractmethod


from autograd import grad, hessian
from autograd.scipy import stats
from autograd.numpy import ndarray
from scipy.optimize import minimize, basinhopping

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.models import CovarianceModel

fftn = np.fft.fftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


class Likelihood(ABC):
    
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, use_taper:None|ndarray=None):
        
        self._z = z
        self.grid = grid
        
        self.use_taper = use_taper
        self.periodogram = Periodogram(taper=use_taper)
        self._I = self.periodogram(z)
        
        self.model = model
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
    
    @abstractmethod
    def update_model_params(self, params: ndarray) -> None:
        free_params = self.model.params        
        updates = dict(zip(free_params.names, params))
        free_params.update_values(updates)
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
    def cov_func(self, params: ndarray, lags:None|list[ndarray, ...] = None) -> list[ndarray, ...]:     # TODO: lags input should be ndarray
        '''compute covariance func on a grid of lags given parameters'''
        
        # TODO: only for dewhittle and whittle
        if lags is None:
            lags = self.grid.lags_unique

        self.update_model_params(params)
        # TODO: ask why ifftshift
        return ifftshift(self.model(lags))
    
    
    @abstractmethod
    def mymethod(self,x:int|ndarray):
        self.abc = x
        return x+5
    
    
    @abstractmethod
    def logprior(self, x: ndarray) -> float:
        '''uninformative prior on the transformed (unrestricted space)'''
        k = self.n_params
        return stats.multivariate_normal.logpdf(x, np.zeros(k), cov=np.eye(k)*100)


    @abstractmethod
    def logpost(self, x: ndarray) -> float:
        return self(x) + self.logprior(x)

    
    @abstractmethod
    def adj_loglik(self, x: ndarray, **loglikargs) -> float: 
        return self(self.MLE.x + self.C @ (x - self.MLE.x), **loglikargs)

    @abstractmethod
    def adj_logpost(self, x: ndarray) -> float:
        return self.adj_loglik(x) + self.logprior(x)

    
    @abstractmethod
    def fit(self, x0: None|ndarray, prior:bool = True, basin_hopping:bool = False, 
                                                       niter:int = 100, 
                                                       print_res:bool = True, **optargs):
        '''fit the model to data - includes optional global optimizer'''
        
        if x0 is None:
            x0 = np.zeros(self.n_params)
            
        
        if prior:                                         # for large samples, the prior is negligible
            attribute = 'MAP'
            def obj(x):     return -self.logpost(x)
            
        else:
            attribute = 'MLE'
            def obj(x):     return -self(x)
            
        if basin_hopping:          # for global optimization
            minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': grad(obj)}
            self.res = basinhopping(obj, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, **optargs) # seed=234230
            success = self.res.lowest_optimization_result['success']
        else:            
            self.res = minimize(x0=x0, fun=obj, jac=grad(obj), method='L-BFGS-B', **optargs)
            success = self.res['success']
            
        if not success:
            print('Optimizer failed!')
            # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
            
        self.res['type'] = attribute
        setattr(self,attribute, self.res)
        
        if attribute=='MLE':
            self.BIC = self.n_params*np.log(self.grid.n_points) - 2*self(self.res.x)         # negative logpost
        
        if print_res:
            print(f'{self.__repr__()} {attribute}:  {np.round(np.exp(self.res.x),3)}')
        
        try:
            self.propcov = np.linalg.inv(-hessian(self.logpost)(self.res.x))
        except np.linalg.LinAlgError:
            print('Singular propcov')
            self.propcov = False
            
        return self.res, self.propcov
    
    
    @abstractmethod
    def sim_MLEs(self, params: ndarray, niter:int=5000, const:str='whittle', **optargs) -> ndarray:
        
        i = 0
        self.MLEs = np.zeros((niter, self.n_params), dtype=np.float64)
        while niter>i: 
            self.update_model_params(np.exp(params))            # list() because of autograd box error
            sampler = SamplerOnRectangularGrid(self.model, self.grid)
            
            _z = sampler()
            _I = self.periodogram(_z)
            
            def obj(x):     return -self.loglik(x, I=_I, const=const)
            _res = minimize(x0=params, fun=obj, jac=grad(obj), method='L-BFGS-B', **optargs)
            if not _res['success']:
                continue
            else:
                print(f'{i+1})   MLE:  {np.round(np.exp(_res.x),3)}')
                self.MLEs[i] = _res.x
                i+=1
            
            self.MLEs_cov = np.cov(self.MLEs.T)
            return self.MLEs


    @abstractmethod
    def estimate_standard_errors_MLE(self):           # maybe not abstract method
        pass
    
    @abstractmethod
    def prepare_curvature_adjustment(self):
        # TODO: singular value decomp
        
        if not hasattr(self, 'MLE'):
            raise TypeError('must optimize log-likelihood first')
            
        B = np.linalg.cholesky(self.MLEs_cov)

        L_inv = np.linalg.inv(np.linalg.cholesky(self.propcov))    # propcov only for MLE
        self.C = np.linalg.inv(B@L_inv)
        
        self.adj_propcov = np.linalg.inv(-hessian(self.adjusted_loglik)(self.MLE.x))
        return
    
    
    
class DebiasedWhittle2(Likelihood):
    
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
        return super().adjusted_loglik(x)
        
    def adj_logpost(self, x: ndarray):
        return super().adjusted_logpost(x)
        
    def mymethod(self, x):
        super().mymethod(x)
    
    def fit(self, x0: None|ndarray, prior:bool = True, basin_hopping:bool = False, 
                                                       niter:int = 100, 
                                                       print_res:bool = True,
                                                       **optargs):
        
        return super().fit(x0=x0, prior=prior, basin_hopping=basin_hopping, niter=niter, print_res=print_res, **optargs)
    
    def sim_MLEs(self, params: ndarray, niter:int=5000, const:str='whittle', **optargs) -> ndarray:
        return super().sim_MLEs(self, params, niter, const, **optargs)
    
    
    def estimate_standard_errors_MLE(self, params: ndarray, monte_carlo:bool=False, niter:int=5000):           # maybe not abstract method
    
        if monte_carlo:
            super().sim_MLEs(params, niter)
        else:
            pass
    
    def prepare_curvature_adjustment(self):           # maybe not abstract method
        return super().prepare_curvature_adjustment()
    
