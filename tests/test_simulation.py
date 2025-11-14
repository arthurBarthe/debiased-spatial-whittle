from debiased_spatial_whittle.sampling.simulation import (
    SamplerOnRectangularGrid,
    MultivariateSamplerOnRectangularGrid,
)
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.grids.base import RectangularGrid


def test_simulation_1d():
    from numpy.random import seed
    from debiased_spatial_whittle.grids.base import RectangularGrid
    from debiased_spatial_whittle.models.univariate import ExponentialModel

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


class TestMultivariateSimulations:
    base_model = ExponentialModel(rho=12)
    bvm = BivariateUniformCorrelation(base_model, r=0.2, f=1.0)

    def test_simulation(self):
        grid = RectangularGrid((64, 64), nvars=2)
        sampler = MultivariateSamplerOnRectangularGrid(self.bvm, grid, p=2)
        assert sampler().shape == (64, 64, 2)
