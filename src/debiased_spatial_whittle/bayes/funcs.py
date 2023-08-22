from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()
from numpy import ndarray



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
