import numpy as np
from numpy.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift
from .spatial_kernel import spatial_kernel
from .models import Parameters


def autocov(cov_func, shape):
    """Compute the covariance function on a grid of lags determined by the passed shape.
    In d=1 the lags would be -(n-1)...n-1, but then a iffshit is applied so that the lags are
    0 ... n-1 -n+1 ... -1. This may look weird but it makes it easier to do the folding operation
    when computing the expecting periodogram"""
    xs = np.meshgrid(*(np.arange(-n + 1, n) for n in shape), indexing='ij')
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
    # We take the real part of the fft only due to numerical precision, in theory this should be real-valued
    result = np.real(fftn(result))
    return result


####NEW OOP VERSION
from .models import CovarianceModel, SeparableModel
from .grids import RectangularGrid


class Periodogram:
    """Class that allows to define a periodogram"""

    def __init__(self, taper = None, scaling='ortho'):
        if taper is None:
            self.taper = lambda x: np.ones_like(x)
        self.scaling = scaling
        #TODO add possibility to not fold?
        #TODO add scaling
        self.fold = True

    def __call__(self, z: np.ndarray):
        # TODO add tapering
        z_taper = z
        f = 1 / z.shape[0] / z.shape[1] * np.abs(fftn(z))**2
        return f


class ExpectedPeriodogram:
    """Class to obtain the expected periodogram when no assumptions are made about the form of the covariance
    model (e.g. separability)."""
    def __init__(self, grid: RectangularGrid, periodogram: Periodogram):
        self.grid = grid
        self.periodogram = periodogram

    @property
    def taper(self):
        return self._taper

    @property
    def periodogram(self):
        return self._periodogram

    @periodogram.setter
    def periodogram(self, value: Periodogram):
        self._periodogram = value
        self._taper = value.taper(self.grid)

    def __call__(self, model: CovarianceModel):
        acv = self.grid.autocov(model)
        return self.compute_ep(acv, self.grid, self.periodogram.fold)

    def gradient(self, model: CovarianceModel, params: Parameters):
        """Provides the derivatives of the expected periodogram with respect to the parameters of the model
        at all frequencies of the Fourier grid. The last dimension is used for different parameters."""
        lags = self.grid.lags_unique
        d_acv = model.gradient(lags, params)
        d_ep = []
        for p_name in params.names:
            aux = ifftshift(d_acv[p_name])
            d_ep.append(self.compute_ep(aux, self.grid, self.periodogram.fold))
        return np.stack(d_ep, axis=-1)


    def compute_ep(self, acv, grid, fold=True):
        """Computes the expected periodogram, for the passed covariance function and grid. The grid is an array, in
        the simplest case just an array of ones
        :param cov_func: covariance function
        :param grid: array
        :param fold: True if we compute the expected periodogram on the natural Fourier grid, False if we compute it on a
        frequency grid with twice higher resolution
        :return: """
        shape = grid.n
        n_dim = grid.ndim
        # In the case of a complete grid, cg takes a closed form.
        cg = spatial_kernel(self.grid.mask)
        # TODO add tapering
        cbar = cg * acv
        # now we need to "fold"
        if fold:
            result = np.zeros(grid.n)
            # TODO make this work for any dimension
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
            result[m + 1:, :n] = cbar[m:, :n]
            result[m + 1:, n + 1:] = cbar[m:, n:]
            result[:m, n + 1:] = cbar[:m, n:]
        # We take the real part of the fft only due to numerical precision, in theory this should be real-valued
        result = np.real(fftn(result))
        return result


class SeparableExpectedPeriodogram(ExpectedPeriodogram):
    """Class to obtain the expected periodogram on a rectangular grid for a separable covariance model,
    in which case separability offers computational gains since the full expected periodogram can
    be computed as the outer product of the expected periodograms in the lower dimensions."""
    # TODO we should ensure the grid is full (or separable for later)

    def __init__(self, grid: RectangularGrid, periodogram: Periodogram):
        super().__init__(grid, periodogram)

    def __call__(self, model: SeparableModel):
        model1, model2 = model.models
        n1, n2 = self.grid.n
        tau1, tau2 = np.arange(n1), np.arange(n2)
        cov_seq1 = model1([tau1, np.zeros_like(tau1)]) * (1 - tau1 / n1)
        cov_seq2 = model2([np.zeros_like(tau2), tau2]) * (1 - tau2 / n2)
        ep1 = 2 * np.real(fft(cov_seq1)).reshape((-1, 1)) - cov_seq1[0]
        ep2 = 2 * np.real(fft(cov_seq2)).reshape((1, -1)) - cov_seq2[0]
        return ep1 * ep2

    def gradient(self, model: SeparableModel):
        """Provides the derivatives of the expected periodogram with respect to the parameters of the model
        at all frequencies of the Fourier grid. The last dimension is used for different parameters."""
        model1, model2 = model.models
        n1, n2 = self.grid.n
        tau1, tau2 = np.arange(n1), np.arange(n2)
        gradient_seq1 = model1.gradient([tau1, ]) * (1 - tau1 / n1)
        gradient_seq2 = model2.gradient([tau2, ]) * (1 - tau2 / n2)
        d_ep1 = 2 * np.real(fft(gradient_seq1, axis=0)).reshape((-1, 1)) - gradient_seq1[0, :]
        d_ep2 = 2 * np.real(fft(gradient_seq2, axis=0)).reshape((1, -1)) - gradient_seq2[0, :]
        return d_ep1 * d_ep2
