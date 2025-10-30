import numpy as np
from debiased_spatial_whittle.grids.base import RectangularGrid


class Sample:
    """
    General class for the definition of a sampled random field. Allows to store computed quantities such as
    periodograms etc.
    """

    def __init__(self, grid: RectangularGrid, values: np.ndarray):
        self.grid = grid
        self.values = values
        self.periodograms = dict()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        eq_grid = self.grid == other.grid
        eq_values = np.all(self.values == other.values)
        return eq_grid and eq_values


class SampleOnRectangularGrid(Sample):
    """
    Class for a sample on a Rectangular grid. In the case of a grid with missing observations, the values at
    missing locations are not used.
    """

    def __init__(self, grid: RectangularGrid, values: np.ndarray):
        assert isinstance(
            grid, RectangularGrid
        ), "The grid should be an instance of RectangularGrid"
        assert values.shape == grid.n, "The shape of the values does not match the grid"
        super(SampleOnRectangularGrid, self).__init__(grid, values)
