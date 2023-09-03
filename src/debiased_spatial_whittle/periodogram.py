from itertools import product
from typing import Tuple

import numpy as np

from .spatial_kernel import spatial_kernel
from .models import Parameters
from .utils import prod_list

fft = np.fft.fft
fftn = np.fft.fftn
ifftshift = np.fft.ifftshift

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
        result = np.zeros_like(grid, dtype=np.complex128)
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
        f = 1 / prod_list(z.shape) * np.abs(fftn(z))**2
        return f


class ExpectedPeriodogram:
    """Class to obtain the expected periodogram when no assumptions are made about the form of the covariance
    model (e.g. separability)."""

    def __init__(self, grid: RectangularGrid, periodogram: Periodogram):
        self.grid = grid
        self.periodogram = periodogram

    @property
    def periodogram(self):
        return self._periodogram

    @periodogram.setter
    def periodogram(self, value: Periodogram):
        self._periodogram = value
        self._taper = value.taper(self.grid)

    @property
    def taper(self):
        return self._taper

    def __call__(self, model: CovarianceModel) -> np.ndarray:
        """

        Parameters
        ----------
        model
            Covariance model under which we compute the expectation of the periodogram

        Returns
        -------
        ep
            The expected periodogram on the grid of Fourier frequencies
        """
        acv = self.grid.autocov(model)
        return self.compute_ep(acv, self.periodogram.fold)

    def gradient(self, model: CovarianceModel, params: Parameters) -> np.ndarray:
        """Provides the gradient of the expected periodogram with respect to the parameters of the model
        at all frequencies of the Fourier grid. The last dimension of the returned array indexes the parameters.

        Parameters
        ----------
        model
            Covariance model. It should implement the gradient method.
        params
            Parameters with which to take the gradient.

        Returns
        -------
        gradient
            Array providing the gradient of the expected periodogram at all Fourier frequencies with respect
            to the requested parameters. The last dimension of the returned array indexes the parameters.
        """
        lags = self.grid.lags_unique
        d_acv = model.gradient(lags, params)
        d_ep = []
        for p_name in params.names:
            aux = ifftshift(d_acv[p_name])
            d_ep.append(self.compute_ep(aux, self.periodogram.fold))
        return np.stack(d_ep, axis=-1)

    def cov_dft_diagonals(self, model: CovarianceModel, m: Tuple[int, int]):
        """Returns the covariance of the DFT over a given diagonal.

        Parameters
        ----------
        model
            Covariance model. The covariance matrix of the DFT will depend on both the model and the sampling.

        m
            Offset in Fourier frequency indices. More precisely, m = (m1, m2) is the offset
            between two frequencies. In dimension 1 this would correspond to i2 - i1 = m1 in terms of
            the indices of Fourier frequencies.

        Returns
        -------
        The covariance of the DFT between Fourier frequencies separated by the offset.
        """
        # TODO only works for 2d
        m1, m2 = m
        n1, n2 = self.grid.n
        acv = self.grid.autocov(model)
        ep = self.compute_ep(acv, d=m)
        return ep[max(0, m1): n1 + m1, max(0, m2): m2 + n2]

    def cov_dft_matrix(self, model: CovarianceModel):
        """
        Provides the covariance matrixof the Discrete Fourier Transform. Computed using matrix products,
        hence not viable for large grid sizes.

        Parameters
        ----------
        model
            Covariance model for which we request the covariance matrix of the Discrete Fourier Transform
        """
        n = self.grid.n
        def transpose(mat):
            mat = np.reshape(mat, (n[0] * n[1], -1))
            mat_t = mat.T
            mat_t = mat_t.reshape((-1, n[0], n[1]))
            return mat_t

        c_x = self.grid.covariance_matrix(model).reshape((-1, n[0], n[1]))
        # applies the multi-dimensional DFT on the rows
        temp = fftn(c_x, axes=(1, 2))
        # flattens out the rows again, transposes, and again reshapes
        temp_T = transpose(temp)
        temp2 = fftn(temp_T.conj(), axes=(1, 2)).conj()
        temp2 = transpose(temp2)
        return temp2 / n[0] / n[1]

    def rel_dft_matrix(self, model: CovarianceModel):
        """
        Provides the relation matrix of the Discrete Fourier Transform. Requires storing the full covariance
        matrix, hencenot viable for large grids. Useful however to check other methods.

        Parameters
        ----------
        model
            Covariance model for which we request the covariance matrix of the Discrete Fourier Transform
        """
        n = self.grid.n

        def transpose(mat):
            mat = np.reshape(mat, (n[0] * n[1], -1))
            mat_t = mat.T
            mat_t = mat_t.reshape((-1, n[0], n[1]))
            return mat_t

        c_x = self.grid.covariance_matrix(model).reshape((-1, n[0], n[1]))
        temp = fftn(c_x, axes=(1, 2))
        temp_T = transpose(temp)
        temp2 = fftn(temp_T, axes=(1, 2))
        temp2 = transpose(temp2)
        return temp2 / n[0] / n[1]


    def cov_diagonals(self, model: CovarianceModel, m: Tuple[int, int]):
        """
        Returns the covariance of the periodogram

        Parameters
        ----------
        model
        m

        Returns
        -------

        """
        return np.abs(self.cov_dft_diagonals(model, m)) ** 2

    def cov_dft_antidiagonals(self, model: CovarianceModel, m: Tuple[int, int]):
        """Returns the covariance of the DFT over a given diagonal. More precisely, m = (m1, m2) is the offset
        between two frequencies. In dimension 1 this would correspond to i2 + i1 = m1 in terms of
        the indices of Fourier frequencies"""
        m1, m2 = m
        n1, n2 = self.grid.n
        acv = self.grid.autocov(model)
        ep = self.compute_ep(acv, d=m)
        return ep[max(m1 - n1 + 1, 0): m1 + 1, max(m2 - n2 + 1, 0): m2 + 1]

    def cov_antidiagonals(self, model: CovarianceModel, m: Tuple[int, int]):
        """

        Parameters
        ----------
        model
        m
        """
        return np.abs(self.cov_dft_antidiagonals(model, m)) ** 2

    def compute_ep(self, acv: np.ndarray, fold: bool = True, d: Tuple[int, int] = (0, 0)):
        """
        Computes the expected periodogram, and more generally any diagonal of the covariance matrix of the Discrete
        Fourier Transform identitied by the two-dimensional offset d. The standard expected periodogram corresponds to
        the default d = (0, 0).

        Parameters
        ----------
        acv
            Autocovariance evaluated on the grid. In the case where the model is multivariate, the first
            dimensions correspond to the lags (one dimension for each spatial dimension), and the last two dimensions
            are used for the covariance matrix.
        fold
            Whether to apply folding of the expected periodogram
        d
            Offset that identifies a hyper-diagonal of the covariance matrix of the DFT.

        Returns
        -------
        np.ndarray
            Expectation of the periodogram
        """
        grid = self.grid
        shape = grid.n
        n_dim = grid.ndim
        # In the case of a complete grid, cg takes a closed form given by the triangle kernel
        if d == (0, 0):
            cg = grid.spatial_kernel
        else:
            cg = spatial_kernel(self.grid.mask, d)
        # TODO add tapering
        # we allow for multivariate, but currently mask same for both grids
        if cg.ndim < acv.ndim:
            cg = np.expand_dims(cg, (-2, -1))
        cbar = cg * acv
        # now we need to "fold"
        if fold:
            #TODO can we go back to complex64?
            # TODO change made specifically for bivariate case. Make general. Should we have just one function for the
            # general case?
            result = np.zeros(grid.n + (2, 2), dtype=np.complex128)
            if n_dim == 2:
                # we could actually always use the general version below but we leave the 2d case as it is easier to
                #read
                for i in range(2):
                    for j in range(2):
                        result[i:, j:, ...] += cbar[i * shape[0]: (i + 1) * shape[0],
                                               j * shape[1]: (j + 1) * shape[1],
                                               ...]
            elif n_dim == 3:
                # we could actually always use the general version below but we leave the 2d case as it is easier to
                #read
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            result[i:, j:, k:] += cbar[i * shape[0]: (i + 1) * shape[0],
                                                  j * shape[1]: (j + 1) * shape[1],
                                                  k * shape[2]: (k + 1) * shape[2]]
            else:
                indexes = product(*[(0, 1) for i_dim in range(n_dim)])
                for ijk in indexes:
                    result[tuple([slice(i, None) for i in ijk])] += \
                        cbar[tuple([slice(i * s, (i + 1) * s) for (i, s) in zip(ijk, shape)])]
        else:
            m, n = shape
            result = np.zeros((2 * m, 2 * n))
            result[:m, :n] = cbar[:m, :n]
            result[m + 1:, :n] = cbar[m:, :n]
            result[m + 1:, n + 1:] = cbar[m:, n:]
            result[:m, n + 1:] = cbar[:m, n:]

        if d == (0, 0):
            # We take the real part of the fft only due to numerical precision, in theory this should be real-valued
            #TODO for multivariate we do not take the real part anymore
            return fftn(result, axes=list(range(n_dim)))
        return fftn(result)


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

    def compute_ep(self, acv: np.ndarray, fold: bool = True, d: Tuple[int, int] = (0, 0)) \
            -> np.ndarray:
        """
        Computes the expected periodogram for the passed finite autocovariance function, in the case where...

        Parameters
        ----------
        acv
        fold
        d

        Returns
        -------

        """