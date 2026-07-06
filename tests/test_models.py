from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()
from numpy.testing import assert_allclose
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.univariate import (
    ExponentialModel,
    SquaredExponentialModel, Matern32Model,
)
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
rand = BackendManager.get_rand()

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
    jac = model.jacobian(g.lags_unique, param_names=(f'{model.name}_rho',))
    g = jac[f'{model.name}_rho']
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
    jac = model.jacobian(g.lags_unique, param_names=(f'{model.name}_rho', f'{model.name}_sigma'))
    g_rho, g_sigma = jac[f'{model.name}_rho'], jac[f'{model.name}_sigma']
    g2 = (acv2 - acv1) / epsilon
    g3 = (acv3 - acv1) / epsilon
    assert_allclose(g_rho, g2, rtol=1e-5, atol=1e-2)
    assert_allclose(g_sigma, g3)


def test_gradient_Matern32():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient, for the Matern32 model.
    """
    g = RectangularGrid((64, 64))
    model = Matern32Model(rho=25, sigma=1)
    epsilon = 1e-7
    acv1 = model(g.lags_unique)
    model.rho = model.rho + epsilon
    acv2 = model(g.lags_unique)
    model.rho = model.rho - epsilon
    model.sigma = model.sigma + epsilon
    acv3 = model(g.lags_unique)
    jac = model.jacobian(g.lags_unique, param_names=(f'{model.name}_rho', f'{model.name}_sigma'))
    g_rho, g_sigma = jac[f'{model.name}_rho'], jac[f'{model.name}_sigma']
    g2 = (acv2 - acv1) / epsilon
    g3 = (acv3 - acv1) / epsilon
    assert_allclose(g_rho, g2, rtol=1e-5, atol=1e-2)
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
    param_name = f'{bvm.name}_r'
    jac = bvm.jacobian(lags, param_names=(param_name,))
    epsilon = 1e-5
    cov = bvm(lags)
    print(param_name)
    setattr(bvm, 'r', getattr(bvm, 'r') + epsilon)
    cov2 = bvm(lags)
    gradient_num = (cov2 - cov) / epsilon
    assert_allclose(jac[param_name], gradient_num, rtol=0.01, atol=1e-2)
    setattr(bvm, 'r', getattr(bvm, 'r') - epsilon)


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
    x1 = rand(25, 3) * 100
    x2 = rand(10, 3) * 100
    mat = model.cov_mat_x1_x2(x1, x2)
    assert mat.ndim == 2
    assert mat.shape == (25, 10)


def test_cov_mat_x1_x2_2():
    model = SquaredExponentialModel()
    model.rho = 10
    model.sigma = 1
    x1 = rand(25, 3) * 100
    mat = model.cov_mat_x1_x2(x1)
    assert mat.ndim == 2
    assert mat.shape == (25, 25)
