from debiased_spatial_whittle.models import BivariateUniformCorrelation, ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.simulation import SamplerBUCOnRectangularGrid
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle
from debiased_spatial_whittle.hypothesis_tests import FixedParametersHT
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.multivariate_periodogram import Periodogram as PeriodogramMulti

def test_fixed_parameters_ht():
    grid = RectangularGrid((64, 64))
    model = ExponentialModel()
    model.rho = 1
    model.sigma = 1
    model.nugget = 0.01
    bvm = BivariateUniformCorrelation(model)
    bvm.r_0 = 0.
    bvm.f_0 = 1.
    sampler = SamplerBUCOnRectangularGrid(bvm, grid)
    z = sampler()

    model.rho = None
    model.sigma = None
    bvm.r_0 = None
    bvm.f_0 = None

    periodogram = PeriodogramMulti()
    expected_periodogram = ExpectedPeriodogram(grid, periodogram)
    dbw = MultivariateDebiasedWhittle(periodogram, expected_periodogram)
    hypothesis_test = FixedParametersHT(bvm, dict(r_0=0.), dbw)
    hypothesis_test(z)

