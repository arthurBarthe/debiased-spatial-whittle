from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

import types
from numpy import ndarray
from autograd import grad, hessian
from numdifftools import Gradient, Hessian
from typing import Callable, Union, List
from scipy.optimize import minimize, basinhopping

def svd_decomp(M: ndarray) -> ndarray:
    '''
    Singular value decomp of a matrix M = U s V^T.

    Parameters
    ----------
    M : ndarray
        matrix.

    Returns
    -------
    ndarray
        L = U sqrt(diag(s)) such has L L^T = M.

    '''
    if any(np.linalg.eigvals(M)<0):
        print(M)
        raise ValueError('Matrix not positive semi-definite?') 
        
    U, s, VT = np.linalg.svd(M)
    L = U @ np.diag(np.sqrt(s))
    if not np.allclose(M, L @ L.T):
        raise ValueError('L L^T != M') 

    return L


def transform(x: ndarray, inv:bool=True) -> ndarray:
    '''
    Transforms parmaeter between the unrestricted and original paramter spaces.

    Parameters
    ----------
    x : ndarray
        vector of parameters.
    inv : bool, optional
        Transform parameters from unrestricted space -> parameter space. The default is True.
        If false, transform parameters from parameter space -> unrestricted space.

    Returns
    -------
    ndarray
        vector of parameters.

    '''
    if inv:
        return np.exp(x)
    
    return np.log(x)


def compute_gradient(func: Callable, 
                     x: ndarray, 
                     approx_grad: bool=False,
                     **func_kwargs) -> ndarray:
    '''
    Computes the gradient of a function evaluated at point x.

    Parameters
    ----------
    func : Callable
        Function for gradient to be computed.
    x : ndarray
        Point which gradient is evaluated.
    approx_grad : bool, optional
        Use numerical differentiation. The default is False.
    **func_kwargs :
        Optional keyword arguments for func.

    Returns
    -------
    ndarray
        Gradient of func at x.

    '''    
    
    if approx_grad:
        grad_f = Gradient(func)(x, **func_kwargs)
    else:
        grad_f = grad(func)(x, **func_kwargs)                 # autodiff
        
        if not np.all(np.isfinite(grad_f)):
            grad_f = -Gradient(func)(x, **func_kwargs)        # fallback to numerical diff

    return grad_f

def compute_hessian(func: Callable, 
                    x: ndarray, 
                    approx_grad: bool=False, 
                    inv:bool=False,
                    **func_kwargs) -> ndarray:
    '''
    Computes the negative Hessian of specified function, usually a 
    log-likelihood or log-posterior.

    Parameters
    ----------
    func : Callable
        Function for Hessian to be computed..
    x : ndarray
        Point which Hessian is evaluated.
    approx_grad : bool, optional
        Use numerical differentiation. The default is False.
    inv : bool, optional
        Compute the inverse of the Hessian. For MCMC proposal covariance matrix. The default is False.
    **func_kwargs :
        Optional keyword arguments for func.

    Raises
    ------
    error
        Prints warning if Hessian not invertible.

    Returns
    -------
    ndarray
        Hessian of func at x.

    '''
    # TODO: test this with compute_propcov_original
    
    if approx_grad:
        hess = -Hessian(func)(x, **func_kwargs)
    else:
        hess = -hessian(func)(x, **func_kwargs)            # autodiff
        
        if not np.all(np.isfinite(hess)):
            hess = -Hessian(func)(x, **func_kwargs)        # fallback to numerical diff
            
    if not inv:
        return hess
        
    try:
        return np.linalg.inv(hess)
    
    except np.linalg.LinAlgError:
        # TODO: raise error?
        print('Singular propcov')
        return False


def fit(func: Callable, 
        x0: Union[None, ndarray]=None,
        basin_hopping:bool = False,
        approx_grad:bool=False,
        transform_params: Union[None,Callable] = None,
        bounds: Union[None, List] = None,
        included_prior: str = False,
        print_res:bool = True,
        save_res:bool=True,
        loglik_kwargs: Union[None,Callable]=None,
        **opt_kwargs):

    '''
    A general optimizer wrapper, includes optional global optimizer
    func is log-likelihood or log-posterior
    '''
    # TODO: test this!
    is_func = isinstance(func, types.FunctionType)    # check if func is a function or class
    
    attribute = 'MAP' if included_prior else 'MLE'     # TODO: not including prior anywhere
    
    if is_func and transform_params is None:
        def transform_params(x: ndarray, inv:bool=True) -> ndarray:    return x
    else:
        transform_params = func.transform     # assumes func has .transform method
        
    if x0 is None and not is_func:
        x0 = transform_params(np.ones(func.n_params), inv=False)
    
    # havent properly tested bounds statements     
    if not is_func and not func.transform_flag and bounds is None:   # assumes func has .transform_lag property
        # print('sdfsdfa')
        bounds = func.model.param_bounds[:len(x0)]       # assumes func has .model.param_bounds method/property
    else:
        bounds=None
    
    if loglik_kwargs is None:
        loglik_kwargs = dict()
        
    def obj(x):     return -func(x, **loglik_kwargs)  # minimize negative
        
    gradient = False if approx_grad else grad(obj)
    print(x0)
    if basin_hopping:          # for global optimization
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient, 'bounds': bounds}
        res = basinhopping(obj, x0, minimizer_kwargs=minimizer_kwargs, **opt_kwargs)   # niter!!
        success = res.lowest_optimization_result['success']
    else:            
        res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', bounds=bounds, **opt_kwargs)
        success = res['success']
        
    if not success:
        print('Optimizer failed!')
        # warnings.warn("Optimizer didn't converge")    # when all warnings are ignored
        
    res['type'] = attribute
    if print_res:
        if not is_func:
            print(f'{func.label} {attribute}:  {transform_params(res.x).round(3)}')
        else:
            print(f'func {attribute}:  {transform_params(res.x).round(3)}')
        
    if save_res and not is_func:
        setattr(func, 'res', res)
    
    # TODO: propcov separate method?
    # self.propcov = self.compute_propcov(res.x, approx_grad)                      
    return res


def RW_MH(niter, x0, log_posterior, propcov, acceptance_lag=1000, **logpost_kwargs):
    from time import time
    n_params  = len(x0)
    h         = 2.38/np.sqrt(n_params)
    
    A          = np.zeros(niter, dtype=np.float64)
    U          = np.random.rand(niter)
    props      = h*np.random.multivariate_normal(np.zeros(len(propcov)), propcov, size=niter)
    post_draws = np.zeros((niter, n_params))
    
    post_draws[0] = x0
    crnt_step     = x0
    bottom        = log_posterior(crnt_step, **logpost_kwargs)
    # print(bottom)
    
    print(f'{"initializing RWMH":-^50}')
    t0 = time()
    for i in range(1, niter):
        
        prop_step = crnt_step + props[i]
        top       = log_posterior(prop_step, **logpost_kwargs)
        # print(top)
        
        A[i]      = np.min((1., np.exp(top-bottom)))
        if U[i] < A[i]:
            crnt_step  = prop_step
            bottom     = top
        
        post_draws[i]   = crnt_step
            
        if (i+1)%acceptance_lag==0:
            print(f'Iteration: {i+1}    Acceptance rate: {A[i-(acceptance_lag-1): (i+1)].mean().round(3)}    Time: {np.round(time()-t0,3)}s')
            
    return post_draws
        
    

from numpy.testing import assert_allclose

def compute_propcov_original(logpost: Callable, x: ndarray, approx_grad: bool=False) -> ndarray:
    '''computes the negative inverse of the hessian evaluated at point x'''
    
    
    if approx_grad:                      # this could be better
        hess = Hessian(logpost)(x)
        return -np.linalg.inv(hess)
    else:    
        try:
            propcov = -np.linalg.inv(hessian(logpost)(x))
            
            if not np.all(np.isfinite(propcov)):       # use numerical diff
                # TODO: test this
                hess = Hessian(logpost)(x)
                propcov = -np.linalg.inv(hess)
            
            return propcov
                
        except np.linalg.LinAlgError:
            # TODO: should we raise error instead?
            print('Singular propcov')
            return False

def test_svd_decomp():
    M = np.eye(4)*np.random.randint(1,10) + np.random.rand()
    L = svd_decomp(M)
    assert_allclose(M, L @ L.T)

if __name__ == '__main__':
    test_svd_decomp()



# def f(x):
#     return 1/x

# print(grad(f)(0.))    

# # def f(x):
# #     return x[0]*x[1]**5

# x = np.array([0.])
# compute_propcov2(f, x, approx_grad=True)
