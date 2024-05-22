from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend('numpy')
np = BackendManager.get_backend()

from numpy.testing import assert_allclose
from debiased_spatial_whittle.periodogram import autocov
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, SeparableExpectedPeriodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, whittle
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, Parameters
from debiased_spatial_whittle.models import BivariateUniformCorrelation


def test_gradient_cov():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient, for the exponential covariance model.
    """
    g = RectangularGrid((64, 64))
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    epsilon = 1e-3
    acv1 = model(g.lags_unique)
    model.rho = model.rho.value + epsilon
    acv2 = model(g.lags_unique)
    g = model.gradient(g.lags_unique, Parameters([model.rho, ]))['rho']
    g2 = (acv2 - acv1) / epsilon
    assert_allclose(g, g2, rtol=1e-3)


def test_gradient_sqExpCov():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient, for the squared exponential covariance
    model.
    """
    g = RectangularGrid((64, 64))
    model = SquaredExponentialModel()
    model.sigma = 1
    model.rho = 25
    model.nugget = 0.01
    epsilon = 1e-7
    acv1 = model(g.lags_unique)
    model.rho = model.rho.value + epsilon
    acv2 = model(g.lags_unique)
    model.rho = model.rho.value - epsilon
    model.sigma = model.sigma.value + epsilon
    acv3 = model(g.lags_unique)
    gradient = model.gradient(g.lags_unique, Parameters([model.rho, model.sigma]))
    g_rho, g_sigma = gradient['rho'], gradient['sigma']
    g2 = (acv2 - acv1) / epsilon
    g3 = (acv3 - acv1) / epsilon
    assert_allclose(g_rho, g2, rtol=1e-5)
    assert_allclose(g_sigma, g3)


def test_gradient_bivariate():
    """
    This test checks that the gradient has the right shape in the case of a bivariate model.
    Returns
    -------

    """
    g = RectangularGrid((32, 32))
    model = SquaredExponentialModel()
    model.rho = 3
    model.sigma = 2
    model.nugget = 0.2
    bvm = BivariateUniformCorrelation(model)
    bvm.r_0 = 0.2
    bvm.f_0 = 1.5
    lags = g.lags_unique
    gradient = bvm.gradient(lags, bvm.params)
    epsilon = 1e-5
    cov = bvm(lags)
    for i, p in enumerate(bvm.params):
        print(p)
        p.value = p.value + epsilon
        cov2 = bvm(lags)
        gradient_num = (cov2 - cov) / epsilon
        assert_allclose(gradient[p.name], gradient_num, rtol=0.01)
        p.value = p.value - epsilon

"""
def test_gradient_cov_separable():
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient, for a separable model.
    rho_0 = 10
    m1 = ExponentialModel()
    m1.rho = rho_0
    m1.sigma = 1
    m2 = ExponentialModel()
    m2.rho = 32
    m2.sigma = 2
    model = SeparableModel((m1, m2), dims=[(0, ), (1, )])
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
"""

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



