import sys

from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

import warnings
from debiased_spatial_whittle.inference.periodogram import autocov
from typing import List

fftn, ifftn = BackendManager.get_fft_methods()
randn = BackendManager.get_randn()
arange = BackendManager.get_arange()


def prod_list(l: List[int]):
    l = list(l)
    if l == []:
        return 1
    else:
        return l[0] * prod_list(l[1:])


# TODO make this work for 1-d and 3-d
def sim_circ_embedding(cov_func, shape):
    cov = autocov(cov_func, shape)
    f = prod_list(shape) * ifftn(cov)
    min_ = np.min(f)
    if min_ < 0:
        sys.exit(0)
        warnings.warn(f"Embedding is not positive definite, min value {min_}.")
    e = np.random.randn(*f.shape) + 1j * np.random.randn(*f.shape)
    z = np.sqrt(np.maximum(f, 0)) * e
    z_inv = 1 / np.sqrt(prod_list(shape)) * np.real(fftn(z))
    for i, n in enumerate(shape):
        z_inv = np.take(z_inv, np.arange(n), i)
    return z_inv, min_

