import numpy as np
from debiased_spatial_whittle import exp_cov, sim_circ_embedding, compute_ep, periodogram
from debiased_spatial_whittle.periodogram import autocov
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, SeparableExpectedPeriodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, whittle
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, ExponentialModelUniDirectional, SeparableModel

def test_oop():
    """
    This test verifies that the oop implementation gives the same debiased whittle likelihood as the old
    implementation.
    :return:
    """
    rho = 10
    rho_lkh = 15
    g = RectangularGrid((512, 512))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = rho
    sampler = SamplerOnRectangularGrid(model, g)
    z = sampler()
    model.rho = rho_lkh
    lkh_oop = d(z, model)
    # old version
    g = np.ones((512, 512))
    cov_func = lambda x: exp_cov(x, rho_lkh)
    e_per = compute_ep(cov_func, g)
    lkh_old = whittle(periodogram(z, g), e_per)
    assert lkh_old == lkh_oop