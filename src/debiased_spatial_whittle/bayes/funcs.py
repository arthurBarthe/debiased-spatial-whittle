from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

from numpy import ndarray
from autograd import grad, hessian
from numdifftools import Hessian
from typing import Callable

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


def compute_propcov(logpost: Callable, x: ndarray, approx_grad: bool=False) -> ndarray:
    # TODO: test this with compute_propcov_original
    '''computes the negative inverse of the hessian evaluated at point x'''
    
    
    if approx_grad:
        hess = Hessian(logpost)(x)
    else:
        hess = hessian(logpost)(x)
        
        if not np.all(np.isfinite(hess)):
            hess = Hessian(logpost)(x)        # fallback to numerical diff
    
    try:
        return -np.linalg.inv(hess)
    
    except np.linalg.LinAlgError:
        # TODO: raise error?
        print('Singular propcov')
        return False


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
        
        
        
    
    
# def f(x):
#     return 1/x

# print(grad(f)(0.))    

# # def f(x):
# #     return x[0]*x[1]**5

# x = np.array([0.])
# compute_propcov2(f, x, approx_grad=True)
