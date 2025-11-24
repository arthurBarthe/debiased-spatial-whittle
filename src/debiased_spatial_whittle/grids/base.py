from functools import cached_property, lru_cache
from debiased_spatial_whittle.models.base import CovarianceModel
from debiased_spatial_whittle.grids.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()


fftn = xp.fft.fftn
ifftn = xp.fft.ifftn
fftshift = xp.fft.fftshift
ifftshift = xp.fft.ifftshift
arange = BackendManager.get_arange()
ones = BackendManager.get_ones()
fftfreq = xp.fft.fftfreq



class RectangularGrid:
    """
    Generic class for hypercubic grids.

    Attributes
    ----------
    n: tuple[int, ...]
        spatial dimensions of the grid in number of points

    delta: tuple[float, ...]
        step sizes of the grid along all dimensions

    nvars: int, optional
        number of variates observed on the grid. Default is 1.

    mask: ndarray
        array of 0's (missing) and 1's (observed) indicating for each point of the grid whether the random field is
        observed at that location.

    n_points: int
        total number of points of the grid

    ndim: int
        number of spatial (or spatio-temporal) dimensions

    extent: tuple[float, ...]
        spatial extent of the grid, determined by the shape and the step sizes

    imshow_extent: tuple[float, ...]
        spatial extent for use as extent parameter of pyplot.imshow

    lags_unique: ndarray
        unique lags of the grid

    grid_points: ndarray
        array with the coordinates of the points of the grid

    lag_matrix: ndarray
        lags between points of the grid

    fourier_frequencies: ndarray
        array of the fourier frequencies of the grid

    fourier_frequencies: ndarray
        array of the fourier frequencies of the grid lags
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        delta: tuple[float, ...] = None,
        mask: xp.ndarray = None,
        nvars: int = 1,
    ):
        """
        Parameters
        ----------
        shape
            shape of the grid

        delta
            grid spacing in all dimensions. If None, set to 1 spatial unit in all dimensions.

        mask
            mask of zero (missing) and ones (observed). If None, set to 1 everywhere.

        nvars
            number of variates of the random field sampled on this grid

        Examples
        --------
        >>> grid_1d = RectangularGrid((64, ), (2., ))
        >>> grid_2d = RectangularGrid((512, 128), (1.5, 2.8))
        >>> grid = RectangularGrid((512, 512), nvars=2)
        """
        self.n = shape
        self.delta = delta
        self.nvars = nvars
        self.mask = mask

    @property
    def n(self) -> tuple[int, ...]:
        """size of the grid"""
        return self._n

    @n.setter
    def n(self, value: tuple[int, ...]):
        self._n = value

    @property
    def ndim(self) -> int:
        """number of spatial dimensions of the grid"""
        return len(self.n)

    @property
    def delta(self) -> tuple[float, ...]:
        """step sizes of the grid along all dimensions, in spatial units."""
        return self._delta

    @delta.setter
    def delta(self, value):
        if value is None:
            value = [
                1.0,
            ] * len(self.n)
        assert len(value) == len(
            self.n
        ), "The length of delta should be equal to the number of dimensions"
        value = tuple([float(v) for v in value])
        self._delta = value

    @property
    def nvars(self) -> int:
        """number of variates. By default, 1, corresponding to a univariate random field."""
        return self._nvars

    @nvars.setter
    def nvars(self, value: int):
        assert isinstance(
            value, int
        ), "The number of components must be integer-valued."
        assert value > 0, "The number of components must be positive."
        self._nvars = value

    @property
    def mask(self) -> xp.ndarray:
        """
        array of 0's (missing) and 1's (observed) indicating for each point of the grid whether the random field is
        observed at that location.
        When nvars is 1 (univariate random field), mask should be an array with ndim dimensions.
        When nvars is greater than one, mask should have an extra dimension, the last one, with size nvars.
        """
        return self._mask

    @mask.setter
    def mask(self, value: xp.ndarray):
        if value is None:
            value = ones(self.n)
            if self.nvars > 1:
                value = ones(self.n + (self.nvars,))
        if self.nvars == 1:
            assert (
                value.shape == self.n
            ), "The shape of the mask should be the same as the shape of the grid"
        else:
            assert value.shape == self.n + (self.nvars,), "Invalid shape of grid mask."
        # TODO for torch we should ensure the mask is on the right device.
        self._mask = value

    @property
    def n_points(self) -> int:
        """total number of points of the grid, irrespective of the mask"""
        return xp.prod(xp.array(self.n))

    @property
    def extent(self) -> tuple[float, ...]:
        """spatial extent of the grid in spatial units"""
        return tuple([n_i * delta_i for n_i, delta_i in zip(self.n, self.delta)])

    @property
    def imshow_extent(self) -> list[float]:
        """extent parameter to pass to matplotlib.pyplot.imshow."""
        extent = self.extent
        imshow_extent = []
        for e in extent:
            imshow_extent.extend((0.0, e))
        return imshow_extent

    @property
    def fourier_frequencies(self) -> xp.ndarray:
        r"""
        Grid of Fourier frequencies corresponding to the spatial grid. For instance, in dimension 1, for
        a grid with $n$ points and a step size $\delta$,
        $
            \left(
                \frac{2k\pi}{\delta n}
            \right)_{k=0, \ldots, n - 1}
            .
        $
        """
        mesh = xp.meshgrid(
            *[fftfreq(n_i, d_i) for n_i, d_i in zip(self.n, self.delta)], indexing="ij"
        )
        return xp.stack(mesh, axis=-1)

    @property
    def fourier_frequencies2(self) -> xp.ndarray:
        r"""
        Grid of Fourier frequencies corresponding to the grid of lags. For instance, in dimension 1, for
        a grid with $n$ points and a step size $\delta$,
        $
            \left(
                \frac{2k\pi}{\delta (2n - 1)}
            \right)_{k=0, \ldots, 2n - 2}
            .
        $
        """
        mesh = xp.meshgrid(
            *[fftfreq(2 * n_i - 1, d_i) for n_i, d_i in zip(self.n, self.delta)],
            indexing="ij",
        )
        out = xp.stack(mesh, axis=-1)
        return BackendManager.convert(out)

    @cached_property
    def lags_unique(self) -> xp.ndarray:
        """shape (2 * n1 + 1, ..., 2 * nd + 1), with d the number of dimensions of the grid."""
        shape = self.n
        delta = self.delta
        lags = xp.meshgrid(
            *(
                arange(-n + 1, n, dtype=xp.float64) * delta_i
                for n, delta_i in zip(shape, delta)
            ),
            indexing="ij",
        )
        return xp.stack(lags, axis=0)

    @property
    def grid_points(self) -> xp.ndarray:
        """list of grid ticks."""
        return tuple(
            [xp.arange(s, dtype=xp.int64) * d for s, d in zip(self.n, self.delta)]
        )

    @cached_property
    def lag_matrix(self) -> xp.ndarray:
        """
        shape (n_points, n_points, n_dimensions),
        matrix of lags between the points of the grids ordered according
        to their coordinates. The matrix may be very large for large grid sizes.
        """
        xs = [xp.arange(s, dtype=xp.int64) * d for s, d in zip(self.n, self.delta)]
        grid = xp.meshgrid(*xs, indexing="ij")
        grid_vec = [g.reshape((-1, 1)) for g in grid]
        lags = [g - g.T for g in grid_vec]
        return xp.array(lags)

    @lru_cache(maxsize=5)
    def spatial_kernel(self, taper_values: xp.ndarray = None):
        """
        Compute the spatial kernel from the grid's mask and the taper values.

        Parameters
        ----------
        taper_values
            Taper values applied to data on the grid

        Returns
        -------
        spatial_kernel
            Shape (2 * n1 - 1, ..., 2 * nd - 1)
        """
        if taper_values is None:
            return spatial_kernel(self.mask, n_spatial_dim=self.ndim)
        return spatial_kernel(self.mask * taper_values.values, n_spatial_dim=self.ndim)

    def covariance_matrix(self, model: CovarianceModel):
        """
        Compute the full covariance matrix under the provided covariance model, while accounting for the mask.

        Parameters
        ----------
        model
            Covariance model

        Returns
        -------
        covmat: ndarray
            shape (n_points, n_points), covariance matrix
        """
        return model(self.lag_matrix) * xp.dot(
            self.mask.reshape((-1, 1)), self.mask.reshape((1, -1))
        )

    def autocov(self, model: CovarianceModel):
        """
        Compute the covariance function on the grid lags.

        Parameters
        ----------
        model: CovarianceModel
            covariance model used

        Returns
        -------
        cov: ndarray
            Shape (2 * n1 - 1, 2 * n2 - 1, ..., 2 * nd - 1) if the grid has shape (n1, ..., nd), see
            notes below.
            Covariance model evaluated on a grid of lags determined by the spatial rectangular grid.

        Notes
        -----
        For instance, in dimension 1, the covariance model is evaluated at lags
            0, 1, ..., n - 1, - n + 1, ..., -1.
        """
        if hasattr(model, "call_on_rectangular_grid"):
            return model.call_on_rectangular_grid(self)
        return ifftshift(model(self.lags_unique), list(range(self.ndim)))

    def autocov_separable(self, model):
        """
        Compute the autocovariance, making use of separability of the model for increased computational efficiency.

        Parameters
        ----------
        model: SeparableModel
            covariance model. Should be separable.

        Returns
        -------
        cov: ndarray
            Autocovariance for the grid's lags.
        """
        assert isinstance(
            model, SeparableModel
        ), "You can only call autocov_separable on a separable model"
        n1, n2 = self.n
        lag1, lag2 = xp.arange(-n1 + 1, n1), xp.arange(-n2 + 1, n2)
        lag1, lag2 = ifftshift(lag1), ifftshift(lag2)
        model1, model2 = model.models
        cov1 = model1(
            [
                lag1,
            ]
        ).reshape((-1, 1))
        cov2 = model2(
            [
                lag2,
            ]
        ).reshape((1, -1))
        return cov1 * cov2

    def separate(self, dims):
        shape = xp.ones(self.ndim, dtype=xp.int64)
        for d in dims:
            shape[d] = self.n[d]
        g = RectangularGrid(shape)
        return g