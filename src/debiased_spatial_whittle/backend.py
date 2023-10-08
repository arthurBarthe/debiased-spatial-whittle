import numpy
import torch

class BackendManager:
    backend_name = 'numpy'
    device = 'cpu'
    
    @classmethod
    def set_backend(cls, name: str):
        cls.backend_name = name
    
    @classmethod
    def get_backend(cls):
        if cls.backend_name == 'numpy':
            return numpy
        elif cls.backend_name == 'autograd':
            import autograd.numpy
            return autograd.numpy
        elif cls.backend_name == 'torch':
            torch.set_default_tensor_type(torch.DoubleTensor)
            torch.array = lambda x: torch.tensor(x, dtype=torch.float64, device=cls.device)
            torch.ndarray = torch.Tensor
            torch.expand_dims = torch.unsqueeze
            torch.take = lambda a, indices, axis: torch.index_select(a, axis, indices)
            torch.pad = lambda a, *args, **kargs: torch.nn.functional.pad(a, args[0][1] + args[0][0], **kargs)
            return torch

    @classmethod
    def get_randn(cls):
        if cls.backend_name == 'numpy':
            return numpy.random.randn
        elif cls.backend_name == 'torch':
            return lambda *args, **kargs: torch.randn(*args, **kargs, dtype=torch.float64, device=cls.device)
        else:
            raise Exception('No backend set')

    @classmethod
    def get_fft_methods(cls):
        if cls.backend_name == 'numpy':
            return numpy.fft.fftn, numpy.fft.ifftn
        elif cls.backend_name == 'torch':
            def new_fftn(a, *args, **kargs):
                kargs['dim'] = kargs['axes']
                del kargs['axes']
                return torch.fft.fftn(a, *args, **kargs)
            fftn = cls._changes_keyword(torch.fft.fftn, 'axes', 'dim')
            ifftn = cls._changes_keyword(torch.fft.ifftn, 'axes', 'dim')
            return fftn, ifftn

    @classmethod
    def _changes_keyword(cls, func, old_keyword, new_keyword):
        def new_func(*args, **kargs):
            if old_keyword in kargs:
                kargs[new_keyword] = kargs[old_keyword]
                del kargs[old_keyword]
            return func(*args, **kargs)
        return new_func