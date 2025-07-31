import numpy as np
from numpy.testing import assert_allclose
from debiased_spatial_whittle.simulation import SamplerBUCOnRectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.models import BivariateUniformCorrelation
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.multivariate_periodogram import Periodogram


class TestEpFullGrid:
    grid = RectangularGrid((64, 64), nvars=2)
    base_model = SquaredExponentialModel(rho=4, sigma=1)
    bvm = BivariateUniformCorrelation(base_model)
    bvm.r = 0.3
    bvm.f = 2.5

    def test_compare_to_average_periodiogram(self):
        sampler = SamplerBUCOnRectangularGrid(self.bvm, self.grid)
        p_computer = Periodogram()
        periodograms = []
        for i in range(10000):
            sample_i = sampler()
            p_value = p_computer([sample_i[..., 0], sample_i[..., 1]])
            periodograms.append(p_value)
        avg_per = np.mean(periodograms, axis=0)
        ep_computer = ExpectedPeriodogram(self.grid, p_computer)
        ep = ep_computer(self.bvm)[:, :, 0, ...]
        # for this model and grid (full grid), the expected periodogram is real-valued
        assert_allclose(np.real(avg_per), np.real(ep), rtol=0.2)
