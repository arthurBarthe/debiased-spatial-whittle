from .backend import BackendManager
np = BackendManager.get_backend()

from functools import cached_property
from abc import ABC, abstractmethod
from pathlib import Path
from functools import cached_property
from typing import Tuple
import matplotlib.pyplot as plt

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
    def __init__(self, shape: Tuple[int], delta: Tuple[float] = None, mask: np.ndarray = None):
        self.n = shape
        self._delta = delta
        self._mask = mask

    @property
    def ndim(self):
        return len(self.n)

    @property
    def delta(self):
        if self._delta is None:
            return [1, ] * len(self.n)
        return self._delta

    @delta.setter
    def delta(self, value):
        assert len(value) == len(self.n), "The length of delta should be equal to the number of dimensions"
        self._delta = value

    @property
    def mask(self):
        if self._mask is None:
            return np.ones(self.n)
        else:
            return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        assert value.shape == self.n, "The shape of the mask should be the same as the shape of the grid"
        self._mask = value

    @property
    def n_points(self):
        """Total number of points of the grid, irrespective of the mask"""
        p = 1
        for ni in self.n:
            p *= ni
        return p

    @cached_property
    def lags_unique(self) -> List[np.ndarray]:
        shape = self.n
        delta = self.delta
        lags = np.meshgrid(*(arange(-n + 1, n) * delta_i for n, delta_i in zip(shape, delta)), indexing='ij')
        return np.stack(lags, axis=0)

    @cached_property
    def lag_matrix(self):
        """
        Matrix of lags between the points of the grids ordered according to their coordinates.

        Returns
        -------
        lags
            shape (n_points, n_points, n_dimensions).
        """
        xs = [np.arange(s, dtype=np.int64) for s in self.n]
        grid = np.meshgrid(*xs, indexing='ij')
        grid_vec = [g.reshape((-1, 1)) for g in grid]
        lags = [g - g.T for g in grid_vec]
        return np.array(lags)

    @property
    def spatial_kernel(self):
        if not hasattr(self, '_spatial_kernel'):
            self._spatial_kernel = spatial_kernel(self.mask)
        return self._spatial_kernel

    def covariance_matrix(self, model: CovarianceModel):
        """
        Compute the full covariance matrix under the provided covariance model, while accounting for the mask.

        Parameters
        ----------
        model
            Covariance model
        """
        return model(self.lag_matrix) * np.dot(self.mask.reshape((-1, 1)), self.mask.reshape((1, -1)))

    def autocov(self, model: CovarianceModel):
        """Compute the covariance function on a grid of lags determined by the passed shape.
        In d=1 the lags would be -(n-1)...n-1, but then a iffshit is applied so that the lags are
        0 ... n-1 -n+1 ... -1. This may look weird but it makes it easier to do the folding operation
        when computing the expecting periodogram"""
        #TODO check that the ifftshift "trick" works for odd sizes
        return ifftshift(model(self.lags_unique))

    def autocov_separable(self, model: SeparableModel):
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