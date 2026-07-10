from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.caching import Freezable, ban_if_frozen

xp = BackendManager.get_backend()


class Sample(Freezable):
    """
    General class for the definition of a sampled random field. Allows to store computed quantities such as
    periodograms etc.
    """

    def __init__(self, grid: RectangularGrid, values: xp.ndarray):
        self.grid = grid
        self.values = values
        super().__init__()
        self.freeze()

    def __array__(self, *args, **kwargs):
        return xp.to_cpu(self.values)


class SampleOnRectangularGrid(Sample):
    """
    Class for a sample on a Rectangular grid. In the case of a grid with missing observations, the values at
    missing locations are not used.
    """

    def __init__(self, grid: RectangularGrid, values: xp.ndarray):
        super(SampleOnRectangularGrid, self).__init__(grid, values)
        assert isinstance(
            grid, RectangularGrid
        ), "The grid should be an instance of RectangularGrid"
        if self.grid.nvars == 1:
            assert values.shape == grid.n, "The shape of the values does not match the grid"
