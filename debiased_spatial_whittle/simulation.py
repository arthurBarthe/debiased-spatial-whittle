import numpy as np
from numpy.fft import fftn, ifftn
import warnings
from .expected_periodogram import autocov
from typing import List

def prod_list(l: List[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])

#TODO make this work for 1-d and 3-d
def sim_circ_embedding(cov_func, shape):
    cov = autocov(cov_func, shape)
    f = np.real(2 ** len(shape) * prod_list(shape) * fftn(cov))
    min_ = np.min(f)
    if min_ <= 0:
        warnings.warn(f'Embedding is not positive definite, min value {min_}.')
    e = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape))
    z = np.sqrt(np.maximum(f, 0)) * e
    z_inv = np.real(ifftn(z))
    for i, n in enumerate(shape):
        z_inv = np.take(z_inv, np.arange(n), i)
    return z_inv, min_



####NEW OOP VERSION
from typing import Tuple
from models import CovarianceModel
from grids import RectangularGrid


def prod_list(l: Tuple[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])


class SamplerOnRectangularGrid:
    """Class that allows to define samplers for Rectangular grids, for which
    fast exact sampling can be achieved via circulant embeddings and the use of the
    Fast Fourier Transform."""

    def __init__(self, model: CovarianceModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self._f = None

    @property
    def f(self):
        if self._f is None:
            cov = self.grid.autocov(self.model)
            f = np.real(2 ** len(self.grid.n) * prod_list(self.grid.n) * fftn(cov))
            min_ = np.min(f)
            if min_ <= 0:
                warnings.warn(f'Embedding is not positive definite, min value {min_}.')
            self._f = f
        return self._f

    # TODO make this work for 1-d and 3-d
    def __call__(self):
        f = self.f
        e = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape))
        z = np.sqrt(np.maximum(f, 0)) * e
        z_inv = np.real(ifftn(z))
        for i, n in enumerate(self.grid.n):
            z_inv = np.take(z_inv, np.arange(n), i)
        return z_inv


