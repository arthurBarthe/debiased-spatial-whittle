__docformat__ = "numpydoc"

from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

from abc import ABC, abstractmethod
from pathlib import Path
from functools import cached_property, lru_cache
from typing import Tuple
import matplotlib.pyplot as plt
from importlib.resources import files

ones = BackendManager.get_ones()

fftfreq = np.fft.fftfreq
from debiased_spatial_whittle.grids.spatial_kernel import spatial_kernel

PATH_TO_FRANCE_IMG = str(files("debiased_spatial_whittle") / "france.jpg")


class Grid(ABC):
    def __init__(self, shape: Tuple[int,]):
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
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        circle = ((x - x_0) ** 2 + (y - y_0) ** 2) <= 1 / 4 * diameter**2
        circle = circle * 1.0
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
        assert 0 <= value <= 1, "p should be a probability"
        self._p = value

    def get_new(self):
        epsilon = np.random.rand(*self.shape)
        return (epsilon >= self.p) * 1.0


class ImgGrid(Grid):
    def __init__(self, shape, img_path: str = PATH_TO_FRANCE_IMG):
        super().__init__(shape)
        self.img_path = img_path
        img = plt.imread(self.img_path)
        img = np.array(img)
        img = (img[110:-110, 110:-110, 0] == 0) * 1.0
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
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return self.img[yy, xx]

    def get_new(self):
        return self.interpolate()
