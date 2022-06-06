import numpy as np
from numpy.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift
from .spatial_kernel import spatial_kernel


def autocov(cov_func, shape):
    """Compute the covariance function on a grid of lags determined by the passed shape.
    In d=1 the lags would be -(n-1)...n-1, but then a iffshit is applied so that the lags are
    0 ... n-1 -n+1 ... -1. This may look weird but it makes it easier to do the folding operation
    when computing the expecting periodogram"""
    xs = np.meshgrid(*(np.arange(-n+1, n) for n in shape), indexing='ij')
    return ifftshift(cov_func(xs))


def compute_ep(cov_func, grid, fold=True):
    """Computes the expected periodogram, for the passed covariance function and grid. The grid is an array, in
    the simplest case just an array of ones
    :param cov_func: covariance function
    :param grid: array
    :param fold: True if we compute the expected periodogram on the natural Fourier grid, False if we compute it on a
    frequency grid with twice higher resolution
    :return: """
    shape = grid.shape
    n_dim = grid.ndim
    # In the case of a complete grid, cg takes a closed form.
    cg = spatial_kernel(grid)
    acv = autocov(cov_func, shape)
    cbar = cg * acv
    # now we need to "fold"
    if fold:
        result = np.zeros_like(grid)
        if n_dim == 1:
            for i in range(2):
                result[i:] += cbar[i * shape[0]: (i + 1) * shape[0]]
        elif n_dim == 2:
            for i in range(2):
                for j in range(2):
                    result[i:, j:] += cbar[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]]
        elif n_dim == 3:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        result[i:, j:, k:] += cbar[i * shape[0]: (i + 1) * shape[0],
                                              j * shape[1]: (j + 1) * shape[1],
                                              k * shape[2]: (k + 1) * shape[2]]
    else:
        m, n = shape
        result = np.zeros((2 * m, 2 * n))
        result[:m, :n] = cbar[:m, :n]
        result[m+1:, :n] = cbar[m:, :n]
        result[m+1:, n+1:] = cbar[m:, n:]
        result[:m, n+1:] = cbar[:m, n:]
    # We take the real part of the fft only due to numerical precision, in theory this should be real
    result = np.real(fftn(result))
    return result
