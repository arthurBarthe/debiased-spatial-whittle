# import autograd.numpy as np
from debiased_spatial_whittle.backend import BackendManager
# TODO: why .backend??
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()


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

def f(x):
    return np.sum(x)*5

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
        
        self.g = np.stack(np.meshgrid(*(np.arange(-n//2,n//2) for n in self.grid.n), indexing='ij'))  # for regular whittle
        
    @property
    def z(self):
        return self._z
    
    @property
    def I(self):
        return self._I
    
    def update_model_params(self, params: ndarray) -> None:
        free_params = self.model.params        
        updates = dict(zip(free_params.names, params))
        free_params.update_values(updates)
        return
        
    def cov_func(self, params: ndarray, lags:None|list[ndarray, ...] = None) -> list[ndarray, ...]:
        '''compute covariance func on a grid of lags given parameters'''
        # TODO: parameter transform is temporary
        
        if lags is None:
            lags = self.grid.lags_unique

        self.update_model_params(params)
        # TODO: ask why ifftshift
        return ifftshift(self.model(lags))
        
        
    def expected_periodogram(self, params: ndarray) -> ndarray:
        acf = self.cov_func(params, lags = None)
        return compute_ep(acf, self.grid.spatial_kernel, self.grid.mask) 
    
    
    def loglik(self, params: ndarray, I:None|ndarray=None, const:str='whittle') -> float:
        # TODO: transform params
        
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
    
    def adjusted_loglik(self, x, **loglikargs): 
        return self.loglik(self.MLE.x + self.C @ (x - self.MLE.x), **loglikargs)
    
    def logprior(self, x: ndarray) -> float:
        '''uninformative prior on the transformed (unrestricted space)'''
        k = self.n_params
        return stats.multivariate_normal.logpdf(x, np.zeros(k), cov=np.eye(k)*100)
    
    
    def logpost(self, x: ndarray) -> float:
        return self.loglik(x) + self.logprior(x)
    
    
    
    def whittle_loglik(self, params: ndarray, I:None|ndarray=None) -> float:
        
        params = np.exp(params)
        
        if I is None:
            I = self.I
            
        acf = ifftshift(self.cov_func(params, self.g))  # ifftshift again?
        f = np.real(fftn(fftshift(acf)))#/(2*np.pi)**2       # this may be unstable for small grids/nugget
        assert np.all(f>0)
        # print(f.min())
        
        ll = -(1/2) * np.sum(np.log(f) + I / f)
        return ll
        
        
    def fit(self, x0: None|ndarray, prior:bool = True, basin_hopping:bool = False, 
                                                       niter:int = 100, 
                                                       label: str = 'debiased Whittle', 
                                                       print_res:bool = True, **optargs):
        '''fit the model to data - includes optional global optimizer'''
        
        if x0 is None:
            x0 = np.zeros(self.n_params)
            
        
        if prior:                                         # for large samples, the prior is negligible
            attribute = 'MAP'
            if label=='whittle':
                def obj(x):     return -self.whittle_loglik(x)
            else:
                def obj(x):     return -self.logpost(x)
        else:
            attribute = 'MLE'
            if label=='whittle':
                def obj(x):     return -self.whittle_loglik(x)
            else:                
                def obj(x):     return -self.loglik(x)
            
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
            self.BIC = self.n_params*np.log(self.grid.n_points) - 2*self.loglik(self.res.x)         # negative logpost
        
        if print_res:
            print(f'{label} {attribute}:  {np.round(np.exp(self.res.x),3)}')
        
        try:
            self.propcov = np.linalg.inv(-hessian(self.logpost)(self.res.x))
        except np.linalg.LinAlgError:
            print('Singular propcov')
            self.propcov = False
            
        return self.res, self.propcov
    
    
    def estimate_standard_errors(self, params: ndarray, monte_carlo:bool=False, niter:int=5000, const:str='whittle', **optargs) -> ndarray:
        
        if monte_carlo:
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
        
    def prepare_curvature_adjustment(self):
        # TODO: singular value decomp
        
        if not hasattr(self, 'MLE'):
            raise TypeError('must optimize log-likelihood first')
            
        B = np.linalg.cholesky(self.MLEs_cov)

        L_inv = np.linalg.inv(np.linalg.cholesky(self.propcov))    # propcov only for MLE
        self.C = np.linalg.inv(B@L_inv)
        
        self.adj_deWhittle_propcov = np.linalg.inv(-hessian(self.adjusted_loglik)(self.MLE.x))
        return
    
    def RW_MH(self, niter:int, posterior_name:str='deWhittle', propcov:None|ndarray=None, acceptance_lag:int=1000, **postargs):
        '''samples from the specified posterior'''
        
        # TODO: mcmc diagnostics
        
        if posterior_name=='deWhittle':
            posterior = self.logpost
            # propcov = 
        elif posterior_name=='adj deWhittle':
            posterior = lambda x: self.adjusted_loglik(x, **postargs)
        # else:
        #     pass
    
        if propcov is None:
            propcov = self.propcov
            
        A     = np.zeros(niter, dtype=np.float64)
        U     = np.random.rand(niter)
            
        h = 2.38/np.sqrt(self.n_params)        
        props = h*np.random.multivariate_normal(np.zeros(self.n_params), propcov, size=niter)
        
        self.post_draws = np.zeros((niter, self.n_params))
        
        crnt_step = self.post_draws[0] = self.res.x
        bottom    = posterior(crnt_step)
        # print(bottom)
        
        print(f'{"initializing RWMH":-^50}')
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