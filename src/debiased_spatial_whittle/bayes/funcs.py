from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

import types
from numpy import ndarray
from autograd import grad, hessian
from numdifftools import Hessian
from typing import Callable, Union
from scipy.optimize import minimize, basinhopping

def svd_decomp(M:ndarray) -> ndarray:
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
    U, s, VT = np.linalg.svd(M)
    return U @ np.diag(np.sqrt(s))


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


def compute_hessian(logpost: Callable, 
                    x: ndarray, 
                    approx_grad: bool=False, 
                    inv:bool=False) -> ndarray:
    
    '''
    computes the negative hessian evaluated at point x
    if inv=True, computes the inverse hessian
    '''    
    # TODO: test this with compute_propcov_original
    
    if approx_grad:
        hess = -Hessian(logpost)(x)
    else:
        hess = -hessian(logpost)(x)            # autodiff
        
        if not np.all(np.isfinite(hess)):
            hess = -Hessian(logpost)(x)        # fallback to numerical diff
            
    if not inv:
        return hess
        
    try:
        return np.linalg.inv(hess)
    
    except np.linalg.LinAlgError:
        # TODO: raise error?
        print('Singular propcov')
        return False


def fit(func: Callable, x0: Union[None, ndarray]=None,
        basin_hopping:bool = False,
        approx_grad:bool=False,
        transform_params: Union[None,Callable] = None,
        include_prior: str = False,
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
    
    attribute = 'MAP' if include_prior else 'MLE'     # TODO: not including prior anywhere
    
    if is_func and transform_params is None:
        def transform_params(x: ndarray, inv:bool=True) -> ndarray:    return x
    else:
        transform_params = func.transform     # assumes func has .transform method
        
    if x0 is None and not is_func:
        x0 = transform_params(np.ones(func.n_params), inv=False)
    
    if loglik_kwargs is None:
        loglik_kwargs = dict()
        
    def obj(x):     return -func(x, **loglik_kwargs)  # minimize negative
        
    gradient = False if approx_grad else grad(obj)
    
    if basin_hopping:          # for global optimization
        minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': gradient}
        res = basinhopping(obj, x0, minimizer_kwargs=minimizer_kwargs, **opt_kwargs)   # niter!!
        success = res.lowest_optimization_result['success']
    else:            
        res = minimize(x0=x0, fun=obj, jac=gradient, method='L-BFGS-B', **opt_kwargs)
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
