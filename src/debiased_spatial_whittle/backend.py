import numpy
import warnings
try:
    import torch
except ModuleNotFoundError:
    warnings.warn('Module Pytorch not found, will not be available as backend.')

try:
    import cupy
except ModuleNotFoundError:
    warnings.warn('Module cupy not found, will not be available as backend.')

def func(x):
    n = len(x)
    res = ()

    for i in range(n):
        res += x[-i - 1]
    return res

def ravel_multi_index(coords, shape):
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).

    Author: Francois Rozet
    (https://github.com/pytorch/pytorch/issues/35674)
    """
    coords = torch.stack(coords, dim=1)
    shape = coords.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


class BackendManager:
    backend_name = 'numpy'
    device = 'cpu'
    
    @classmethod
    def set_backend(cls, name: str):
        cls.backend_name = name
    
    @classmethod
    def get_backend(cls):
        if cls.backend_name == 'numpy':
            numpy.to_cpu = lambda x: x
            return numpy
        elif cls.backend_name == 'cupy':
            cupy.to_cpu = lambda x: x.get()
            return cupy
        elif cls.backend_name == 'autograd':
            import autograd.numpy
            return autograd.numpy
        elif cls.backend_name == 'torch':
            torch.to_cpu = lambda x: x.cpu()
            torch.set_default_tensor_type(torch.DoubleTensor)
            torch.array = lambda x: torch.tensor(x, dtype=torch.float64, device=cls.device)
            torch.ndarray = torch.Tensor
            torch.expand_dims = torch.unsqueeze
            torch.take = lambda a, indices, axis: torch.index_select(a, axis, indices)
            torch.pad = lambda a, *args, **kargs: torch.nn.functional.pad(a, func(args[0]), **kargs)
            torch.dot = torch.matmul
            torch.digitize = torch.bucketize
            torch.ravel_multi_index = ravel_multi_index
            torch.Tensor.astype = lambda self, type: self.to(dtype=type)
            return torch

    @classmethod
    def convert(cls, a):
        if BackendManager.backend_name == 'torch':
            return torch.tensor(a).to(device=BackendManager.device)
        else:
            return a

    @classmethod
    def get_zeros(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.zeros
        if cls.backend_name == 'cupy':
            return cupy.zeros
        elif cls.backend_name == 'torch':
            return lambda *args, **kargs: torch.zeros(*args, **kargs, device=BackendManager.device)

    @classmethod
    def get_randn(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.random.randn
        elif cls.backend_name == 'cupy':
            return cupy.random.randn
        elif cls.backend_name == 'torch':
            return lambda *args, **kargs: torch.randn(*args, **kargs, dtype=torch.float64, device=cls.device)
        else:
            raise Exception('No backend set')

    @classmethod
    def get_arange(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.arange
        elif cls.backend_name == 'cupy':
            return cupy.arange
        elif cls.backend_name == 'torch':
            return lambda *args, **kargs: torch.arange(*args, **kargs, device=cls.device)
        else:
            raise Exception('No backend set')

    @classmethod
    def get_slogdet(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.linalg.slogdet
        elif cls.backend_name == 'cupy':
            return cupy.linalg.slogdet
        elif cls.backend_name == 'torch':
            return torch.linalg.slogdet
        else:
            raise Exception('No backend set')

    @classmethod
    def get_inv(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.linalg.inv
        elif cls.backend_name == 'cupy':
            return cupy.linalg.inv
        elif cls.backend_name == 'torch':
            return torch.linalg.inv
        else:
            raise Exception('No backend set')

    @classmethod
    def get_fft_methods(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.fft.fftn, numpy.fft.ifftn
        elif cls.backend_name == 'cupy':
            return cupy.fft.fftn, cupy.fft.ifftn
        elif cls.backend_name == 'torch':
            def new_fftn(a, *args, **kargs):
                kargs['dim'] = kargs['axes']
                del kargs['axes']
                return torch.fft.fftn(a, *args, **kargs)
            fftn = cls._changes_keyword(torch.fft.fftn, 'axes', 'dim')
            ifftn = cls._changes_keyword(torch.fft.ifftn, 'axes', 'dim')
            return fftn, ifftn

    @classmethod
    def get_fftshift_methods(cls):
        if cls.backend_name == 'numpy' or cls.backend_name == 'autograd':
            return numpy.fft.fftshift, numpy.fft.ifftshift
        elif cls.backend_name == 'cupy':
            return cupy.fft.fftshift, cupy.fft.ifftshift
        elif cls.backend_name == 'torch':
            fftshift = cls._changes_keyword(torch.fft.fftshift, 'axes', 'dim')
            ifftshift = cls._changes_keyword(torch.fft.ifftshift, 'axes', 'dim')
            return fftshift, ifftshift

    @classmethod
    def _changes_keyword(cls, func, old_keyword, new_keyword):
        def new_func(*args, **kargs):
            if old_keyword in kargs:
                kargs[new_keyword] = kargs[old_keyword]
                del kargs[old_keyword]
            return func(*args, **kargs)
        return new_func

    @classmethod
    def ensure_array(cls, a):
        if cls.backend_name == 'torch':
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a)
            return a.to(device=cls.device)
        else:
            return a
