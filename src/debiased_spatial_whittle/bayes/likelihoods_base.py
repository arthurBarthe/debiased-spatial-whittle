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
    
    @property
    def n_points(self):
        return self.grid.n_points
    
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
    def cov_func(self, params: ndarray, lags:None|list[ndarray, ...] = None) -> list[ndarray, ...]:
        '''compute covariance func on a grid of lags given parameters'''
        
        # TODO: only for dewhittle and whittle
        if lags is None:
            lags = self.grid.lags_unique

        self.update_model_params(params)
        # TODO: ask why ifftshift
        return ifftshift(self.model(np.stack(lags)))
    
    
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
                                                       niter:int = 100, approx_grad:bool=False,
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
            
        if not approx_grad:
            gradient = grad(obj)
        else:
            gradient = False
            
        if basin_hopping:          # for global optimization
            minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient}
            self.res = basinhopping(obj, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, **optargs)
            success = self.res.lowest_optimization_result['success']
        else:            
            self.res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **optargs)
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
        
        if approx_grad:
            self.propcov = self.res.hess_inv.todense()
        else:    
            try:
                self.propcov = np.linalg.inv(-hessian(self.logpost)(self.res.x))
            except np.linalg.LinAlgError:
                print('Singular propcov')
                self.propcov = False
                
        return self.res, self.propcov
    
    
    @abstractmethod
    def sim_MLEs(self, params: ndarray, niter:int=5000, t_random_field:bool=False, df:None|int=10, **optargs) -> ndarray:
        
        i = 0
        self.MLEs = np.zeros((niter, self.n_params), dtype=np.float64)
        while niter>i:
            
            _z = self.sim_z(np.exp(params))
            # TODO: put t_random_field in sampler()
            if t_random_field:
                if df == np.inf:
                    chi = np.ones(self.n_points)
                else:
                    chi = np.random.chisquare(df, self.n_points)/df
                
                _z /= np.sqrt(chi.reshape(self.grid.n))
                
                if False:
                    import matplotlib.pyplot as plt
                    plt.imshow(_z)
                    plt.show()
                
            _I = self.periodogram(_z)
            
            def obj(x):     return -self(x, I=_I)    # TODO: likelihood_kwargs, e.g. const
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
        
        self.adj_propcov = np.linalg.inv(-hessian(self.adj_loglik)(self.MLE.x))
        return
    
    @abstractmethod
    def sim_z(self,params:None|ndarray=None):
        if params is None:
            params = np.exp(self.res.x)
            
        self.update_model_params(params)            # list() because of autograd box error
        sampler = SamplerOnRectangularGrid(self.model, self.grid)
        return sampler()
    
    
    @abstractmethod
    def RW_MH(self, niter:int, adjusted:bool=False, acceptance_lag:int=1000, **postargs):
        '''Random walk Metropolis-Hastings: samples the specified posterior'''
        
        # TODO: mcmc diagnostics
        
        if adjusted:
            posterior = self.adj_logpost
            propcov   = self.adj_propcov
            label = 'adjusted ' + self.__repr__()
            
        else:
            posterior = self.logpost
            propcov   = self.propcov
            label = self.__repr__()
    
        A     = np.zeros(niter, dtype=np.float64)
        U     = np.random.rand(niter)
            
        h = 2.38/np.sqrt(self.n_params)        
        props = h*np.random.multivariate_normal(np.zeros(self.n_params), propcov, size=niter)
        
        self.post_draws = np.zeros((niter, self.n_params))
        
        crnt_step = self.post_draws[0] = self.res.x
        bottom    = posterior(crnt_step)
        # print(bottom)
        
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