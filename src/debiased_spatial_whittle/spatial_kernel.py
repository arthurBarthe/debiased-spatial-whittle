import autograd.numpy as np
from autograd.numpy.fft import fftn, ifftn
from typing import Tuple
# TODO: have to adjust imports from backend

def spatial_kernel(g: np.ndarray, m: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Compute the spatial kernel, cg in the paper, via FFT for computational efficiency.

    Parameters
    ----------
    g
        mask of observations, or more generally pointwise modulation e.g. a taper or the product of a taper with an
        observation mask
    m
        offset in frequency indices

    Returns
    -------
    cg
        Spatial kernel
    """
    n = g.shape
    two_n = tuple([s * 2 - 1 for s in n])
    if m == (0, 0):
        f = np.abs(fftn(g, two_n))**2
        cg = ifftn(f)
        cg /= np.sum(g ** 2)
        return cg
    # TODO only works in 2d right now
    m1, m2 = m
    n1, n2 = n
    a = np.exp(2j * np.pi * m1 / n1 * np.arange(n1)).reshape((-1, 1))
    a = a * np.exp(2j * np.pi * m2 / n2 * np.arange(n2)).reshape((1, -1))
    g2 = g * a
    f = fftn(g, two_n) * np.conj(fftn(g2, two_n))
    cg = ifftn(f)
    cg /= np.sum(g**2)
    return cg
