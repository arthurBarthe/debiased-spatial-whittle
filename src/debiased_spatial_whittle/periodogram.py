from .backend import BackendManager

np = BackendManager.get_backend()

from typing import Tuple

from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.utils import prod_list

fft = np.fft.fft
fftn = np.fft.fftn
ifftshift = np.fft.ifftshift
ndarray = np.ndarray


def autocov(cov_func, shape):
    """Compute the covariance function on a grid of lags determined by the passed shape.
    In d=1 the lags would be -(n-1)...n-1, but then a iffshit is applied so that the lags are
    0 ... n-1 -n+1 ... -1. This may look weird but it makes it easier to do the folding operation
    when computing the expecting periodogram"""
    xs = np.meshgrid(*(np.arange(-n + 1, n) for n in shape), indexing="ij")
    if BackendManager.backend_name == "torch":
        # TODO this is a temporary solution, not ideal though
        xs = xs.to(device=BackendManager.device)
    return ifftshift(cov_func(xs))


def compute_ep(
    acf: ndarray, spatial_kernel: ndarray, grid: ndarray = None, fold: bool = True
) -> ndarray:
    """Computes the expected periodogram, for the passed covariance function and grid. The grid is an array, in
    the simplest case just an array of ones
    :param cov_func: covariance function
    :param grid: array
    :param fold: True if we compute the expected periodogram on the natural Fourier grid, False if we compute it on a
    frequency grid with twice higher resolution
    :return:"""
    shape = grid.shape
    n1, n2 = grid.shape
    n_dim = grid.ndim
    # In the case of a complete grid, spatial_kernel takes a closed form.
    cbar = spatial_kernel * acf

    # now we need to "fold"
    if fold:
        # TODO: make this autograd compatible for any d with any n's
        result = np.zeros(shape, dtype=np.complex128)

        if n_dim == 1:
            for i in range(2):
                res = cbar[i * shape[0] : (i + 1) * shape[0]]
                result += np.pad(res, (i, 0), mode="constant")

        elif n_dim == 2:
            for i in range(2):
                for j in range(2):
                    res = cbar[
                        i * shape[0] : (i + 1) * shape[0],
                        j * shape[1] : (j + 1) * shape[1],
                    ]
                    result += np.pad(
                        res, ((i, 0), (j, 0)), mode="constant"
                    )  # autograd solution

        elif n_dim == 3:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        res = cbar[
                            i * shape[0] : (i + 1) * shape[0],
                            j * shape[1] : (j + 1) * shape[1],
                            k * shape[2] : (k + 1) * shape[2],
                        ]
                        result += np.pad(res, ((i, 0), (j, 0), (k, 0)), mode="constant")

    else:
        m, n = shape
        result = np.zeros((2 * m, 2 * n))
        result[:m, :n] = cbar[:m, :n]
        result[m + 1 :, :n] = cbar[m:, :n]
        result[m + 1 :, n + 1 :] = cbar[m:, n:]
        result[:m, n + 1 :] = cbar[:m, n:]
    # We take the real part of the fft only due to numerical precision, in theory this should be real
    result = np.real(fftn(result))
    return result


def compute_ep_old(cov_func, grid, fold=True):
    """Computes the expected periodogram, for the passed covariance function and grid. The grid is an array, in
    the simplest case just an array of ones
    :param cov_func: covariance function
    :param grid: array
    :param fold: True if we compute the expected periodogram on the natural Fourier grid, False if we compute it on a
    frequency grid with twice higher resolution
    :return:"""
    shape = grid.shape
    n_dim = grid.ndim
    # In the case of a complete grid, cg takes a closed form.
    cg = spatial_kernel(grid)
    acv = autocov(cov_func, shape)
    cbar = cg * acv
    # now we need to "fold"
    if fold:
        result = np.zeros_like(grid)
        result = np.zeros_like(grid, dtype=np.complex128)
        if n_dim == 1:
            for i in range(2):
                result[i:] += cbar[i * shape[0] : (i + 1) * shape[0]]
        elif n_dim == 2:
            for i in range(2):
                for j in range(2):
                    result[i:, j:] += cbar[
                        i * shape[0] : (i + 1) * shape[0],
                        j * shape[1] : (j + 1) * shape[1],
                    ]
        elif n_dim == 3:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        result[i:, j:, k:] += cbar[
                            i * shape[0] : (i + 1) * shape[0],
                            j * shape[1] : (j + 1) * shape[1],
                            k * shape[2] : (k + 1) * shape[2],
                        ]
    else:
        m, n = shape
        result = np.zeros((2 * m, 2 * n))
        result[:m, :n] = cbar[:m, :n]
        result[m + 1 :, :n] = cbar[m:, :n]
        result[m + 1 :, n + 1 :] = cbar[m:, n:]
        result[:m, n + 1 :] = cbar[:m, n:]
    # We take the real part of the fft only due to numerical precision, in theory this should be real-valued
    result = np.real(fftn(result))
    return result


####NEW OOP VERSION
from typing import Union
from debiased_spatial_whittle.models import CovarianceModel, ModelParameter
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.samples import SampleOnRectangularGrid

ones = BackendManager.get_ones()


class Periodogram:
    """
    Provides the capability to compute the periodogram of the data.

    Attributes
    ----------
    taper: function handle
        tapering function

    fold: boolean
        Whether to fold the periodogram.
    """

    def __init__(self, taper=None):
        if taper is None:
            self.taper = lambda shape: ones(shape)
        self.fold = True
        self._version = 0

    def __hash__(self):
        return id(self) + self._version

    @property
    def fold(self):
        """Whether to compute a folded version of the periodogram."""
        return self._fold

    @fold.setter
    def fold(self, value: bool):
        self._fold = value

    def __call__(self, sample: Union[np.ndarray, SampleOnRectangularGrid]):
        """
        Computes the periodogram of the data.

        Parameters
        ----------
        sample: ndarray | SampleOnRectangularGrid
            Sampled data on the grid. Can either be an ndarray, or an instance of SampleOnRectangularGrid.
            In the latter case, repeated calls to this method will access cached values of the periodogram
            rather than carrying out the same computation again.

        Returns
        -------
        periodogram: ndarray
            Periodogram of the data
            - shape (2 * n1 + 1, ..., 2 * nk + 1) if the fold attribute is False
            - shape (n1, ..., nk) if the fold attribute is True
        """
        if isinstance(sample, SampleOnRectangularGrid):
            if self in sample.periodograms:
                return sample.periodograms[self]
            else:
                z_values = sample.values * self.taper(sample.grid.n)
        else:
            z_values = sample * self.taper(sample.shape)
        f = 1 / prod_list(z_values.shape) * np.abs(fftn(z_values)) ** 2
        if isinstance(sample, SampleOnRectangularGrid):
            sample.periodograms[self] = f
        return f

    def __setattr__(self, key, value):
        """
        Sets attribute and update version of the object, which will update its hash, so that stored periodogram
        values are not used if properties of the periodogram are changed.

        Parameters
        ----------
        key
            name of the attribute
        value
            value of the attribute
        """
        if "_version" in self.__dict__:
            self.__dict__["_version"] += 1
        super(Periodogram, self).__setattr__(key, value)


class HashableArray:
    def __init__(self, values: np.array):
        self.values = values

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self.values == other.values


class ExpectedPeriodogram:
    r"""
    Provides the capability to compute the expected periodogram on a fixed grid for
    any covariance model.

    Attributes
    ----------
    grid: RectangularGrid
        sampling grid

    periodogram: Periodogram
        periodogram for which we require the expectation. This is necessary to account for tapering for instance.

    Notes
    -----
    In dimension 1, the formula for the expected periodogram can be written as,

    $$
        \overline{I}(\omega_k) =
        \sum_{\tau=-n + 1}^{n - 1}
        c_g(\tau)
        c_X(\tau)
        e^{i \omega_k \tau},
    $$

    or equivalently,

    $$
        \overline{I}(\omega_k) =
        \sum_{\tau=0}^{2n - 1}
        \left[
            c_g(\tau)
            c_X(\tau)
            +
            c_g(- n + \tau)
            c_X(- n + \tau)
        \right]
        e^{i \omega_k \tau}.
    $$

    The latter can naturally be implemented via FFT and generalizes to higher-dimension domains.
    """

    def __init__(self, grid: RectangularGrid, periodogram: Periodogram):
        self.grid = grid
        self.periodogram = periodogram

    @property
    def grid(self) -> RectangularGrid:
        """Sampling grid"""
        return self._grid

    @grid.setter
    def grid(self, value: RectangularGrid):
        self._grid = value

    @property
    def periodogram(self):
        """periodogram for which we require the expectation."""
        return self._periodogram

    @periodogram.setter
    def periodogram(self, value: Periodogram):
        self._periodogram = value
        if self.grid.nvars == 1:
            self._taper = HashableArray(value.taper(self.grid.n))
        else:
            self._taper = HashableArray(value.taper(self.grid.n + (self.grid.nvars,)))

    @property
    def taper(self):
        return self._taper

    def __call__(self, model: CovarianceModel) -> np.ndarray:
        """
        Compute the expected periodogram for this covariance model.

        Parameters
        ----------
        model
            Covariance model under which we compute the expectation of the periodogram

        Returns
        -------
        ep: ndarray
            Shape  (n1, n2, ..., nk).
            The expected periodogram on the grid of Fourier frequencies.

            If the fold attribute of the periodogram is False, the shape of the returned array is instead
            (2 * n1 + 1, ..., 2 * nk + 1).
        """
        acv = self.grid.autocov(model)
        return self.compute_ep(acv, self.periodogram.fold)

    def _reshape_acv(self, acv):
        """
        TODO: NOT USED. TO BE DELETED
        Method for internal call that ensures acv has the right number of dimensions before applying the algorithm
        that computes the expected periodogram. Specifically, if d is the number of spatial dimensions, p the number
        of variates (including p=1, in which case acv might typically have (n1, ..., nd) or (n1, ..., nd, m)
        for m model parameters), then the returned shape should be (n1, ..., nd, m, p, p). In the multivariate case
        acv will have shape (n1, ..., nd, p, p) or (n1, ..., nd, m, p, p). Again the returned shape should be
        (n1, ..., nd, m, p, p)
        """
        ndim = self.grid.ndim
        p = self.grid.nvars
        acv_shape, acv_ndim = acv.shape, acv.ndim
        if acv.ndim == ndim:
            return np.reshape(acv, acv_shape + (1, 1, 1))
        if acv.ndim == ndim + 1:
            return np.reshape(acv, acv_shape + (1, 1))

    def compute_ep(
        self,
        acv: np.ndarray,
        fold: bool = True,
        d: Tuple[int, int] = (0, 0),
        apply_cg: bool = True,
    ):
        """
        Computes the expected periodogram, and more generally any diagonal of the covariance matrix of the Discrete
        Fourier Transform identitied by the two-dimensional offset d. The standard expected periodogram corresponds to
        the default d = (0, 0).

        Parameters
        ----------
        acv: ndarray
            Autocovariance evaluated on the grid's lags. For a grid with shape (n1, ..., nd), the first d dimensions
            of acv should have sizes (2 * n1 - 1, ..., 2 * nd - 1).
            The standard way to obtain acv is through the call of the autocov method of a rectangular grid.
            acv may have extra dimensions. The following cases are standard:
            1. Univariate data, multiple model parameter vectors. acv will have shape (2 * n1 - 1, ..., 2 * nd - 1, m)
            where m is the number of model parameter vectors.
            2. Multivariate data, unique model parameter vector. acv will have shape
            (2 * n1 - 1, ..., 2 * nd - 1, p, p) where p is the number of variates
            3. Multivariate data, multiple model parameter vectors. acv will have shape
            (2 * n1 - 1, ..., 2 * nd - 1, m, p, p)
        fold
            Whether to apply folding of the expected periodogram
        d
            Offset that identifies a hyper-diagonal of the covariance matrix of the DFT.

        Returns
        -------
        ep: np.ndarray
            Expectation of the periodogram.

            - shape (2 * n1 - 1, ..., 2 * nd - 1) is the fold attribute of self.periodogram is False

            - shape (n1, ..., nd) if fold is True

        Notes
        -----
        For standard use cases, this should not be called directly. Instead, one should directly call
        the __call__ method.
        """
        grid = self.grid
        shape = grid.n
        n_dim = grid.ndim
        p = grid.nvars
        # In the case of a complete grid, cg takes a closed form given by the triangle kernel
        if d == (0, 0):
            cg = grid.spatial_kernel(self.taper)
        else:
            cg = spatial_kernel(self.grid.mask, d)
        if p == 1:
            cg = np.reshape(cg, cg.shape + (1,) * (acv.ndim - n_dim))
        else:
            cg = np.reshape(
                cg, cg.shape[:n_dim] + (1,) * (acv.ndim - n_dim - 2) + (p, p)
            )
        cbar = acv
        if apply_cg:
            cbar = cg * acv
        # now we need to "fold" the spatial dimensions
        zeros_ = ((0, 0),) * (acv.ndim - n_dim)
        if fold:
            if BackendManager.backend_name == "torch":
                result = np.zeros(
                    shape + acv.shape[n_dim:],
                    dtype=np.complex128,
                    device=BackendManager.device,
                )
            else:
                result = np.zeros(shape + acv.shape[n_dim:], dtype=np.complex128)
            if n_dim == 1:
                for i in range(2):
                    res = cbar[i * shape[0] : (i + 1) * shape[0]]
                    result += np.pad(res, ((i, 0),) + zeros_, mode="constant")

            elif n_dim == 2:
                for i in range(2):
                    for j in range(2):
                        res = cbar[
                            i * shape[0] : (i + 1) * shape[0],
                            j * shape[1] : (j + 1) * shape[1],
                        ]
                        result += np.pad(
                            res,
                            (
                                (i, 0),
                                (j, 0),
                            )
                            + zeros_,
                            mode="constant",
                        )  # autograd solution

            elif n_dim == 3:
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            res = cbar[
                                i * shape[0] : (i + 1) * shape[0],
                                j * shape[1] : (j + 1) * shape[1],
                                k * shape[2] : (k + 1) * shape[2],
                            ]
                            result += np.pad(
                                res,
                                (
                                    (i, 0),
                                    (j, 0),
                                    (k, 0),
                                )
                                + zeros_,
                                mode="constant",
                            )

            # else:
            #     indexes = product(*[(0, 1) for i_dim in range(n_dim)])
            #     for ijk in indexes:
            #         result[tuple([slice(i, None) for i in ijk])] += \
            #             cbar[tuple([slice(i * s, (i + 1) * s) for (i, s) in zip(ijk, shape)])]
        else:
            m, n = shape
            result = np.zeros((2 * m, 2 * n))
            result[:m, :n] = cbar[:m, :n]
            result[m + 1 :, :n] = cbar[m:, :n]
            result[m + 1 :, n + 1 :] = cbar[m:, n:]
            result[:m, n + 1 :] = cbar[:m, n:]

        if d == (0, 0):
            out = fftn(result, None, list(range(n_dim)))
            if grid.nvars == 1:
                out = np.real(out)
            return out
        out = fftn(result)
        if grid.nvars == 1:
            out = np.reshape(out, grid.n)
        return out

    def gradient(self, model: CovarianceModel, params: list[ModelParameter]) -> ndarray:
        """
        Provides the gradient of the expected periodogram with respect to the parameters of the model
        at all frequencies of the Fourier grid. The last dimension of the returned array indexes the parameters.

        Parameters
        ----------
        model: CovarianceModel
            Covariance model. It should implement the gradient method.

        params: Parameters
            Parameters with which to take the gradient.

        Returns
        -------
        gradient: ndarray
            Array providing the gradient of the expected periodogram at all Fourier frequencies with respect
            to the requested parameters. The last dimension of the returned array indexes the parameters.

        Notes
        -----
        This requires that the model's _gradient method be implemented.
        """
        lags = self.grid.lags_unique
        d_acv = model.gradient(lags, params)
        aux = ifftshift(d_acv, list(range(lags.shape[0])))
        return self.compute_ep(aux, self.periodogram.fold)

    def cov_dft_matrix(self, model: CovarianceModel):
        r"""
        Provides the complex-valued covariance matrix of the Discrete Fourier Transform.
        Implemented via FFT, but still requires the full covariance matrix of the data,
        hence this is not viable for large grids.

        Parameters
        ----------
        model: CovarianceModel
            Covariance model for which we request the covariance matrix of the Discrete Fourier Transform

        Returns
        -------
        cov_dft: ndarray (n1, n2, ..., nd)
            Covariance matrix of the Discrete Fourier Transform of the data.

        Notes
        -----
        The result of this method corresponds to,

        $$
            E[\mathbf{J} \mathbf{J}^*] = E[U^* \mathbf{X} \mathbf{X}^* U]=U^* E[\mathbf{X} \mathbf{X}^T] U= U^* C_X U
        $$

        where $\mathbf{J}$ is the Discrete Fourier Transform (DFT) of the data $\mathbf{X}$,
        whose covariance matrix is denoted by $C_X$, and $U$ is the DFT matrix.
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
        matrix, hence not viable for large grids. Useful however to check other methods.

        Parameters
        ----------
        model: CovarianceModel
            Covariance model for which we request the covariance matrix of the Discrete Fourier Transform

        Returns
        -------
        rel_dft: ndarray (n1, n2, ..., nk)
            Relation matrix of the Discrete Fourier Transform of the data
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
        cov_dft: ndarray
            The covariance of the DFT between Fourier frequencies separated by the offset.

        Notes
        -----
        When m is zero everywhere, this is just the expected periodogram.
        """
        # TODO only works for 2d
        m1, m2 = m
        n1, n2 = self.grid.n
        acv = self.grid.autocov(model)
        ep = self.compute_ep(acv, d=m)
        return ep[max(0, m1) : n1 + m1, max(0, m2) : m2 + n2]

    def cov_diagonals(self, model: CovarianceModel, m: Tuple[int, int]):
        """
        Returns the covariance of the periodogram (valid only in 2d).

        Parameters
        ----------
        model: CovarianceModel
            True covariance model

        m: tuple[int, int]
            frequency offset

        Returns
        -------
        cov: ndarray
            Covariance of the periodogram between frequencies offset by m
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
        return ep[max(m1 - n1 + 1, 0) : m1 + 1, max(m2 - n2 + 1, 0) : m2 + 1]

    def cov_antidiagonals(self, model: CovarianceModel, m: Tuple[int, int]):
        """

        Parameters
        ----------
        model
        m
        """
        return np.abs(self.cov_dft_antidiagonals(model, m)) ** 2


class SeparableExpectedPeriodogram(ExpectedPeriodogram):
    """Class to obtain the expected periodogram on a rectangular grid for a separable covariance model,
    in which case separability offers computational gains since the full expected periodogram can
    be computed as the outer product of the expected periodograms in the lower dimensions."""

    # TODO we should ensure the grid is full (or separable for later)

    def __init__(self, grid: RectangularGrid, periodogram: Periodogram):
        super().__init__(grid, periodogram)

    def __call__(self, model):
        model1, model2 = model.models
        n1, n2 = self.grid.n
        tau1, tau2 = np.arange(n1), np.arange(n2)
        cov_seq1 = model1(
            [
                tau1,
            ]
        ) * (1 - tau1 / n1)
        cov_seq2 = model2(
            [
                tau2,
            ]
        ) * (1 - tau2 / n2)
        ep1 = 2 * np.real(fft(cov_seq1)).reshape((-1, 1)) - cov_seq1[0]
        ep2 = 2 * np.real(fft(cov_seq2)).reshape((1, -1)) - cov_seq2[0]
        return ep1 * ep2

    def gradient(self, model):
        """Provides the derivatives of the expected periodogram with respect to the parameters of the model
        at all frequencies of the Fourier grid. The last dimension is used for different parameters."""
        model1, model2 = model.models
        n1, n2 = self.grid.n
        tau1, tau2 = np.arange(n1), np.arange(n2)
        gradient_seq1 = model1.gradient(
            [
                tau1,
            ]
        ) * (1 - tau1 / n1)
        gradient_seq2 = model2.gradient(
            [
                tau2,
            ]
        ) * (1 - tau2 / n2)
        d_ep1 = (
            2 * np.real(fft(gradient_seq1, axis=0)).reshape((-1, 1))
            - gradient_seq1[0, :]
        )
        d_ep2 = (
            2 * np.real(fft(gradient_seq2, axis=0)).reshape((1, -1))
            - gradient_seq2[0, :]
        )
        return d_ep1 * d_ep2

    def compute_ep(
        self, acv: np.ndarray, fold: bool = True, d: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
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
        raise NotImplementedError("This has not been implemented yet.")
