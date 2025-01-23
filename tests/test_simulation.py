from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid


def test_simulation_1d():
    from numpy.random import seed
    from debiased_spatial_whittle.grids import RectangularGrid
    from debiased_spatial_whittle.models import ExponentialModel

    seed(1712)
    model = ExponentialModel()
    model.rho = 2
    model.sigma = 1
    grid1 = RectangularGrid((16, 1))
    grid2 = RectangularGrid((16,))
    sampler1 = SamplerOnRectangularGrid(model, grid1)
    sampler2 = SamplerOnRectangularGrid(model, grid2)
    z1 = sampler1()
    seed(1712)
    z2 = sampler2()
    assert True


class TestSingleSimulation:
    model = ExponentialModel()
    model.rho = 10
    model.sigma = 1
    grid = RectangularGrid((256, 256))

    def test_simulation(self):
        sampler = SamplerOnRectangularGrid(self.model, self.grid)
        sampler()


class TestMultipleSimulations:
    model = ExponentialModel()
    model.rho = 10
    model.sigma = 1
    grid = RectangularGrid((256, 256))

    def test_simulation(self):
        sampler = SamplerOnRectangularGrid(self.model, self.grid)
        sampler.n_sims = 36
        z = sampler()
        assert z.shape == self.grid.n

    def test_independent(self):
        # TODO add this test to check that realizations are i.i.d.
        pass
