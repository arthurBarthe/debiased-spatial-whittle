__docformat__ = "numpydoc"

from debiased_spatial_whittle.backend import BackendManager
np = BackendManager.get_backend()

from abc import ABC, abstractmethod
from pathlib import Path
from functools import cached_property, lru_cache
from typing import Tuple
import matplotlib.pyplot as plt


fftfreq = np.fft.fftfreq
from debiased_spatial_whittle.spatial_kernel import spatial_kernel

PATH_TO_FRANCE_IMG = str(Path(__file__).parents[2] / 'france.jpg')


class Grid(ABC):
    def __init__(self, shape: Tuple[int]):
        self.shape = shape

    @abstractmethod
    def get_new(self):
        pass

    def __mul__(self, other):
        return GridProduct(self, other)


class GridProduct(Grid):
    def __init__(self, grid1, grid2):
        assert grid1.shape == grid2.shape
        shape = grid1.shape
        super(GridProduct, self).__init__(shape)
        self.grid1 = grid1
        self.grid2 = grid2

    def get_new(self):
        return self.grid1.get_new() * self.grid2.get_new()


class FullGrid(Grid):
    def get_new(self):
        return np.ones(self.shape)


class CircleGrid(Grid):
    def __init__(self, shape: Tuple[int], center: Tuple[int], diameter: float):
        super().__init__(shape)
        self.center = center
        self.diameter = diameter

    def get_new(self):
        (x_0, y_0), diameter = self.center, self.diameter
        shape = self.shape
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        circle = ((x - x_0) ** 2 + (y - y_0) ** 2) <= 1 / 4 * diameter ** 2
        circle = circle * 1.
        return circle


class BernoulliGrid(Grid):
    def __init__(self, shape: Tuple[int], p: float):
        super(BernoulliGrid, self).__init__(shape)
        self.p = p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value: float):
        assert 0 <= value <= 1, 'p should be a probability'
        self._p = value

    def get_new(self):
        epsilon = np.random.rand(*self.shape)
        return (epsilon >= self.p) * 1.


class ImgGrid(Grid):
    def __init__(self, shape, img_path: str = PATH_TO_FRANCE_IMG):
        super().__init__(shape)
        self.img_path = img_path
        img = plt.imread(self.img_path)
        img = np.array(img)
        img = (img[110:-110, 110:-110, 0] == 0) * 1.
        self.img = np.flipud(img)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self._img = value

    def interpolate(self):
        m_0, n_0 = self.img.shape
        m, n = self.shape
        x = np.asarray(np.arange(n) / n * n_0, dtype=np.int64)
        y = np.asarray(np.arange(m) / m * m_0, dtype=np.int64)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        return self.img[yy, xx]

    def get_new(self):
        return self.interpolate()



###NEW OOP VERSION
from debiased_spatial_whittle.models import CovarianceModel, SeparableModel
from typing import List, Tuple

fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
arange = BackendManager.get_arange()


class RectangularGrid:
    """
    Generic class for hypercubic grids.

    Attributes
    ----------
    n: tuple[int]
        spatial dimensions of the grid in number of points

    delta: tuple[float]
        step sizes of the grid along all dimensions

    nvars: int, optional
        number of variates observed on the grid. Default is 1.

    mask: ndarray
        array of 0's (missing) and 1's (observed) indicating for each point of the grid whether the random field is
        observed at that location.
        When nvars is 1 (univariate random field), mask should be an array with len(n) dimensions.
        When nvars is greater than one, mask should have an extra dimension, the last one, with size nvars.
    """
    def __init__(self, shape: tuple[int], delta: tuple[float] = None, mask: np.ndarray = None, nvars: int = 1):
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
    def n(self) -> tuple[int]:
        """size of the grid"""
        return self._n

    @n.setter
    def n(self, value: tuple[int]):
        self._n = value

    @property
    def ndim(self) -> int:
        """number of spatial dimensions of the grid"""
        return len(self.n)

    @property
    def delta(self) -> tuple[float]:
        """step sizes of the grid along all dimensions, in spatial units."""
        return self._delta

    @delta.setter
    def delta(self, value):
        if value is None:
            value = [1., ] * len(self.n)
        assert len(value) == len(self.n), "The length of delta should be equal to the number of dimensions"
        value = tuple([float(v) for v in value])
        self._delta = value

    @property
    def nvars(self) -> int:
        """number of variates. By default, 1, corresponding to a univariate random field."""
        return self._nvars

    @nvars.setter
    def nvars(self, value: int):
        assert isinstance(value, int), "The number of components must be integer-valued."
        assert value > 0, "The number of components must be positive."
        self._nvars = value

    @property
    def mask(self) -> np.ndarray:
        """observation mask. By default, 1 everywhere."""
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        if value is None:
            value = np.ones(self.n)
            if self.nvars > 1:
                value = np.ones(self.n + (self.nvars,))
        if self.nvars == 1:
            assert value.shape == self.n, "The shape of the mask should be the same as the shape of the grid"
        else:
            assert value.shape == self.n + (self.nvars, ), "Invalid shape of grid mask."
        self._mask = value

    @property
    def n_points(self):
        """int: Total number of points of the grid, irrespective of the mask"""
        return np.prod(np.array(self.n))

    @property
    def extent(self) -> tuple[float]:
        """spatial extent of the grid in spatial units"""
        return tuple([n_i * delta_i for n_i, delta_i in zip(self.n, self.delta)])

    @property
    def imshow_extent(self) -> list[tuple[float]]:
        """extent parameter to pass to matplotlib.pyplot.imshow."""
        extent = self.extent
        imshow_extent = []
        for e in extent:
            imshow_extent.extend((0, e))
        return imshow_extent

    @property
    def fourier_frequencies(self) -> np.ndarray:
        """Grid of Fourier frequencies corresponding to the spatial grid."""
        mesh = np.meshgrid(*[fftfreq(n_i, d_i) for n_i, d_i in zip(self.n, self.delta)])
        return np.stack(mesh, axis=-1)

    @property
    def fourier_frequencies2(self) -> np.ndarray:
        """Grid of Fourier frequencies corresponding to the spatial grid, without folding."""
        mesh = np.meshgrid(*[fftfreq(2 * n_i - 1, d_i) for n_i, d_i in zip(self.n, self.delta)])
        out = np.stack(mesh, axis=-1)
        return BackendManager.convert(out)

    @cached_property
    def lags_unique(self) -> List[np.ndarray]:
        """ndarray: dtype float64, shape (2 * n1 + 1, ..., 2 * nk + 1), with k is the number of dimensions of the grid."""
        shape = self.n
        delta = self.delta
        lags = np.meshgrid(*(arange(-n + 1, n, dtype=np.float64) * delta_i for n, delta_i in zip(shape, delta)), indexing='ij')
        return np.stack(lags, axis=0)

    @property
    def grid_points(self) -> np.ndarray:
        """dtype float, list of grid ticks."""
        return tuple([np.arange(s, dtype=np.int64) * d for s, d in zip(self.n, self.delta)])

    @cached_property
    def lag_matrix(self) -> np.ndarray:
        """
        shape (n_points, n_points, n_dimensions),
        matrix of lags between the points of the grids ordered according
        to their coordinates. The matrix may be very large for large grid sizes.
        """
        xs = [np.arange(s, dtype=np.int64) * d for s, d in zip(self.n, self.delta)]
        grid = np.meshgrid(*xs, indexing='ij')
        grid_vec = [g.reshape((-1, 1)) for g in grid]
        lags = [g - g.T for g in grid_vec]
        return np.array(lags)

    @lru_cache(maxsize=5)
    def spatial_kernel(self, taper_values: np.ndarray = None):
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
        return model(self.lag_matrix) * np.dot(self.mask.reshape((-1, 1)), self.mask.reshape((1, -1)))

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
            Covariance model evaluated on a grid of lags determined by the spatial rectangular grid.

        Notes
        -----
        For instance, in dimension 1, the covariance model is evaluated at lags
            0, 1, ..., n - 1, - n + 1, ..., -1.
        """
        if hasattr(model, 'call_on_rectangular_grid'):
            return model.call_on_rectangular_grid(self)
        return ifftshift(model(self.lags_unique), list(range(self.ndim)))

    def autocov_separable(self, model: SeparableModel):
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
        assert isinstance(model, SeparableModel), "You can only call autocov_separable on a separable model"
        n1, n2 = self.n
        lag1, lag2 = np.arange(-n1 + 1, n1), np.arange(-n2 + 1, n2)
        lag1, lag2 = ifftshift(lag1), ifftshift(lag2)
        model1, model2 = model.models
        cov1 = model1([lag1, ]).reshape((-1, 1))
        cov2 = model2([lag2, ]).reshape((1, -1))
        return cov1 * cov2

    def separate(self, dims):
        shape = np.ones(self.ndim, dtype=np.int64)
        for d in dims:
            shape[d] = self.n[d]
        g = RectangularGrid(shape)
        return g
