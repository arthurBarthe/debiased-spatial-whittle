import numpy as np
from numpy.fft import fftn, ifftn


def spatial_kernel(data):
    """Compute the spatial kernel, cg in the paper, via FFT.

    :param data: np.ndarray, the data for which we want to compute the sample autocovariance
    In the case of a full grid with g=1 everywhere on the grid this takes a simple analytical form."""
    n = list(data.shape)
    two_n = list(data.shape)
    for i in range(len(two_n)):
        two_n[i] = two_n[i] * 2 - 1
    f = abs(fftn(data, two_n))**2
    slices = tuple(slice(None, 2 * s - 1, None) for s in n)
    cg = np.real(ifftn(f)[slices])
    # Normalise as in the paper
    cg /= cg.flatten()[0]
    return cg
