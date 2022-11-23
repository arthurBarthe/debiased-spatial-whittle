import numpy as np
from numpy.testing import assert_allclose
from debiased_spatial_whittle import exp_cov, sim_circ_embedding, compute_ep, periodogram
from debiased_spatial_whittle.periodogram import autocov
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, SeparableExpectedPeriodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, whittle
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, ExponentialModelUniDirectional, SeparableModel, Parameters


def test_gradient_cov():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient.
    """
    g = RectangularGrid((64, 64))
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    epsilon = 1e-3
    acv1 = model(g.lags_unique)
    model.rho = 10 + epsilon
    acv2 = model(g.lags_unique)
    g = model.gradient(g.lags_unique, Parameters([model.rho, ]))['rho']
    g2 = (acv2 - acv1) / epsilon
    assert_allclose(g, g2, rtol=1e-3)


def test_gradient_cov_separable():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient, for a separable model.
    """
    rho_0 = 10
    m1 = ExponentialModelUniDirectional(axis=0)
    m1.rho = rho_0
    m1.sigma = 1
    m2 = ExponentialModelUniDirectional(axis=1)
    m2.rho = 32
    m2.sigma = 2
    model = SeparableModel((m1, m2))
    # simulation
    g = RectangularGrid((128, 128))
    acv1 = model(g.lags_unique)
    epsilon = 1e-3
    m1.rho = 10 + epsilon
    acv2 = model(g.lags_unique)
    g = model.gradient(g.lags_unique, Parameters([m1.rho, ]))
    g = g['rho_0']
    g2 = (acv2 - acv1) / epsilon
    assert_allclose(g, g2, rtol=1e-2)


def test_gradient_cov_merged_params():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient, in the case where two parameters are merged into one.
    """
    grid = RectangularGrid((64, 64))
    model = ExponentialModel()
    model.merge_parameters(('rho', 'sigma'))
    model.sigma = 5
    g = model.gradient(grid.lags_unique, Parameters([model.rho, ]))['rho and sigma']
    epsilon = 1e-3
    acv1 = model(grid.lags_unique)
    model.sigma = 5 + epsilon
    acv2 = model(grid.lags_unique)
    g2 = (acv2 - acv1) / epsilon
    assert_allclose(g, g2, rtol=1e-2)