from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    MaternModel,
)
from debiased_spatial_whittle.grids import RectangularGrid


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

    def test_independent(self):
        # TODO add this test to check that realizations are i.i.d.
        pass
