import numpy as np
from numpy.fft import fftn, ifftn
import warnings
from .expected_periodogram import autocov


def sim_circ_embedding(cov_func, shape):
    cov = autocov(cov_func, shape)
    m, n = shape
    f = np.real(4 * m * n * fftn(cov))
    min_ = np.min(f)
    if min_ <= 0:
        warnings.warn(f'Embedding is not positive definite, min value {min_}.')
    e = (np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape))
    z = np.sqrt(np.maximum(f, 0)) * e
    z_inv = ifftn(z)
    return np.real(z_inv[:shape[0], :shape[1]]), min_
