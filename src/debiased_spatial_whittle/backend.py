import numpy
import warnings


import scipy.special

TORCH_INSTALLED = True
CUPY_INSTALLED = True

try:
    import torch
except ModuleNotFoundError:
    TORCH_INSTALLED = False

try:
    import cupy
    from cupy_backends.cuda.api.runtime import CUDARuntimeError
    import cupyx.scipy.special
except ModuleNotFoundError:
    CUPY_INSTALLED = False


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
    backend_name = "numpy"
    device = "cpu"
    block = False

    @classmethod
    def list_avail(cls):
        """list available backends. As Cupy can only be used on GPU, it will be listed as available only if the package
        is installed and a GPU device is available"""
        if TORCH_INSTALLED:
            print("Torch CPU available")
        if TORCH_INSTALLED and torch.cuda.is_available():
            print("Torch GPU available")
        if CUPY_INSTALLED:
            try:
                cupy.zeros(1)
                print("CUPY GPU available")
            except CUDARuntimeError as e:
                pass

    @classmethod
    def set_backend(cls, name: str):
        if cls.block:
            # raise an error as the backend should be set before usage
            raise Exception(
                f"Cannot change backend (current: {cls.backend_name}). Make sure you set the backend first thing."
            )
        if name == "cupy" and (not CUPY_INSTALLED):
            warnings.warn(
                "Module Cupy not found, will not be available as backend. Falling back to numpy."
            )
            name = "numpy"
        elif name == "torch" and (not TORCH_INSTALLED):
            warnings.warn(
                "Module Torch not found, will not be available as backend. Falling back to numpy"
            )
            name = "numpy"
        cls.backend_name = name

    @classmethod
    def get_backend(cls):
        cls.block = True
        if cls.backend_name == "numpy":
            numpy.to_cpu = lambda x: x
            numpy.item = lambda x: x
            return numpy
        elif cls.backend_name == "cupy":
            cupy.to_cpu = lambda x: x.get()
            cupy.item = lambda x: x.item()
            return cupy
        elif cls.backend_name == "autograd":
            import autograd.numpy

            return autograd.numpy
        elif cls.backend_name == "torch":
            torch.to_cpu = lambda x: x.cpu()
            torch.item = lambda x: x.item()
            torch.set_default_tensor_type(torch.DoubleTensor)
            torch.array = lambda x: torch.tensor(
                x, dtype=torch.float64, device=cls.device
            )
            torch.ndarray = torch.Tensor
            torch.expand_dims = torch.unsqueeze
            torch.take = lambda a, indices, axis: torch.index_select(a, axis, indices)
            torch.pad = lambda a, *args, **kargs: torch.nn.functional.pad(
                a, func(args[0]), **kargs
            )
            torch.dot = torch.matmul
            torch.digitize = torch.bucketize
            torch.ravel_multi_index = ravel_multi_index
            torch.Tensor.astype = lambda self, type: self.to(dtype=type)
            return torch

    @classmethod
    def convert(cls, a):
        if BackendManager.backend_name == "torch":
            return torch.tensor(a).to(device=BackendManager.device)
        elif BackendManager.backend_name == "numpy":
            return a
        elif BackendManager.backend_name == "cupy":
            return cupy.asarray(a)

    @classmethod
    def get_zeros(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.zeros
        if cls.backend_name == "cupy":
            return cupy.zeros
        elif cls.backend_name == "torch":
            return lambda *args, **kargs: torch.zeros(
                *args, **kargs, device=BackendManager.device
            )

    @classmethod
    def get_ones(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.ones
        if cls.backend_name == "cupy":
            return cupy.ones
        elif cls.backend_name == "torch":
            return lambda *args, **kargs: torch.ones(
                *args, **kargs, device=BackendManager.device
            )

    @classmethod
    def get_randn(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.random.randn
        elif cls.backend_name == "cupy":
            return cupy.random.randn
        elif cls.backend_name == "torch":
            return lambda *args, **kargs: torch.randn(
                *args, **kargs, dtype=torch.float64, device=cls.device
            )
        else:
            raise Exception("No backend set")

    @classmethod
    def get_arange(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.arange
        elif cls.backend_name == "cupy":
            return cupy.arange
        elif cls.backend_name == "torch":
            return lambda *args, **kargs: torch.arange(
                *args, **kargs, device=cls.device
            )
        else:
            raise Exception("No backend set")

    @classmethod
    def get_slogdet(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.linalg.slogdet
        elif cls.backend_name == "cupy":
            return cupy.linalg.slogdet
        elif cls.backend_name == "torch":
            return torch.linalg.slogdet
        else:
            raise Exception("No backend set")

    @classmethod
    def get_inv(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.linalg.inv
        elif cls.backend_name == "cupy":
            return cupy.linalg.inv
        elif cls.backend_name == "torch":
            return torch.linalg.inv
        else:
            raise Exception("No backend set")

    @classmethod
    def get_fft_methods(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.fft.fftn, numpy.fft.ifftn
        elif cls.backend_name == "cupy":
            return cupy.fft.fftn, cupy.fft.ifftn
        elif cls.backend_name == "torch":

            def new_fftn(a, *args, **kargs):
                kargs["dim"] = kargs["axes"]
                del kargs["axes"]
                return torch.fft.fftn(a, *args, **kargs)

            fftn = cls._changes_keyword(torch.fft.fftn, "axes", "dim")
            ifftn = cls._changes_keyword(torch.fft.ifftn, "axes", "dim")
            return fftn, ifftn

    @classmethod
    def get_fftshift_methods(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return numpy.fft.fftshift, numpy.fft.ifftshift
        elif cls.backend_name == "cupy":
            return cupy.fft.fftshift, cupy.fft.ifftshift
        elif cls.backend_name == "torch":
            fftshift = cls._changes_keyword(torch.fft.fftshift, "axes", "dim")
            ifftshift = cls._changes_keyword(torch.fft.ifftshift, "axes", "dim")
            return fftshift, ifftshift

    @classmethod
    def get_gamma(cls):
        if cls.backend_name == "numpy" or cls.backend_name == "autograd":
            return scipy.special.gamma
        elif cls.backend_name == "cupy":
            return cupyx.scipy.special.gamma
        elif cls.backend_name == "torch":
            lgamma = torch.lgamma

            def torch_gamma(*args, **kwargs):
                x, *args = args
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(
                        [
                            x,
                        ]
                    ).to(device=cls.device)
                return torch.exp(lgamma(x, *args, **kwargs))

            return torch_gamma

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
        if cls.backend_name == "torch":
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a)
            return a.to(device=cls.device)
        else:
            return a

    @classmethod
    def to_cpu(cls, a):
        if cls.backend_name == "numpy":
            return a
        if cls.backend_name == "cupy":
            return a.get()
        if cls.backend_name == "numpy":
            return a.cpu()
