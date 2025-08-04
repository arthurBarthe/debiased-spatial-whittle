from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")
np = BackendManager.get_backend()
from numpy.testing import assert_allclose
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
)
from debiased_spatial_whittle.models import BivariateUniformCorrelation


def test_model():
    model = SquaredExponentialModel(rho=12, sigma=1)
    assert model(np.arange(10.0).reshape(1, -1)).shape == (10,)


def test_model_array():
    rhos = np.array([12.0, 15.0])
    sigmas = np.array([1.0, 1.0])
    model = SquaredExponentialModel(rho=rhos, sigma=sigmas)
    assert model(np.arange(10.0).reshape(1, -1)).shape == (10, 2)


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
    model.rho = model.rho + epsilon
    acv2 = model(g.lags_unique)
    g = model.gradient(
        g.lags_unique,
        [
            model.param.rho,
        ],
    )[..., 0]
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
    epsilon = 1e-7
    acv1 = model(g.lags_unique)
    model.rho = model.rho + epsilon
    acv2 = model(g.lags_unique)
    model.rho = model.rho - epsilon
    model.sigma = model.sigma + epsilon
    acv3 = model(g.lags_unique)
    gradient = model.gradient(g.lags_unique, [model.param.rho, model.param.sigma])
    g_rho, g_sigma = gradient[..., 0], gradient[..., 1]
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
    g = RectangularGrid((32, 32), nvars=2)
    model = SquaredExponentialModel(rho=3.0, sigma=1.2)
    bvm = BivariateUniformCorrelation(model, r=0.2, f=0.1)
    lags = g.lags_unique
    params_for_gradient = [
        bvm.param.r,
    ]
    gradient = bvm.gradient(lags, params_for_gradient)
    epsilon = 1e-5
    cov = bvm(lags)
    for i, p in enumerate(params_for_gradient):
        print(p.name)
        setattr(bvm, p.name, getattr(bvm, p.name) + epsilon)
        cov2 = bvm(lags)
        gradient_num = (cov2 - cov) / epsilon
        assert_allclose(gradient[..., i], gradient_num[:, :, 0, ...], rtol=0.01)
        setattr(bvm, p.name, getattr(bvm, p.name) - epsilon)


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


def test_cov_mat_x1_x2():
    model = SquaredExponentialModel(rho=10, sigma=1)
    x1 = np.random.rand(25, 3) * 100
    x2 = np.random.rand(10, 3) * 100
    mat = model.cov_mat_x1_x2(x1, x2)
    assert mat.ndim == 2
    assert mat.shape == (25, 10)


def test_cov_mat_x1_x2_2():
    model = SquaredExponentialModel()
    model.rho = 10
    model.sigma = 1
    x1 = np.random.rand(25, 3) * 100
    mat = model.cov_mat_x1_x2(x1)
    assert mat.ndim == 2
    assert mat.shape == (25, 25)
