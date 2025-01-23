from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("autograd")
np = BackendManager.get_backend()

import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    MaternModel,
    MaternCovarianceModel,
    Parameters,
)
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import (
    DeWhittle,
    Whittle,
    Gaussian,
    MCMC,
    GaussianPrior,
    Optimizer,
)
from debiased_spatial_whittle.simulation import (
    SamplerOnRectangularGrid,
    SquaredSamplerOnRectangularGrid,
)

from debiased_spatial_whittle.bayes.funcs import transform
from numpy.testing import assert_allclose

from autograd import grad, hessian, jacobian

np.random.seed(5325234)
inv = np.linalg.inv


def test_likelihood_dewhittle_grad():
    n = (64, 64)
    params = np.array([8.0, 1.0])
    model = SquaredExponentialModel()
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = 0.1
    grid = RectangularGrid(n)

    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    dw = DeWhittle(
        z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform
    )

    x = np.log(params)
    ll_grad = grad(dw)(x)
    ll_hess = hessian(dw)(x)
    assert np.all(np.isfinite(ll_grad))
    assert np.all(np.isfinite(ll_hess))


def test_likelihood_whittle_grad():
    n = (64, 64)
    params = np.array([8.0, 1.0])
    model = SquaredExponentialModel()
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = 0.1
    grid = RectangularGrid(n)

    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    whittle = Whittle(
        z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform
    )

    x = np.log(params)
    ll_grad = grad(whittle)(x)
    ll_hess = hessian(whittle)(x)
    assert np.all(np.isfinite(ll_grad))
    assert np.all(np.isfinite(ll_hess))


def test_likelihood_gauss_grad():
    n = (16, 16)
    params = np.array([2.0, 1.0])
    model = SquaredExponentialModel()
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = 0.1
    grid = RectangularGrid(n)

    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    gauss = Gaussian(
        z, grid, SquaredExponentialModel(), nugget=0.1, transform_func=transform
    )

    x = np.log(params)
    ll_grad = grad(gauss)(x)
    ll_hess = hessian(gauss)(x)  # TODO: this is slow!
    assert np.all(np.isfinite(ll_grad))
    assert np.all(np.isfinite(ll_hess))


def test_dewhittle_d_cov_func():
    """test of derivates of covariances function, original and reparameterized version, only for SqExp and Exp models"""
    n = (16, 16)
    grid = RectangularGrid(n)
    lags = grid.lags_unique
    model = ExponentialModel()  # SquaredExponentialModel()

    dw = DeWhittle(np.ones(n), grid, model, nugget=0.1, transform_func=None)
    params = np.array([8.0, 2.0]) + np.random.rand(2)
    grads = jacobian(dw.cov_func)(params, lags).T
    assert_allclose(grads, dw.d_cov_func(params))

    def reparamed_cov(x, lags):
        return dw.cov_func(transform(x, inv=True), lags)

    dw = DeWhittle(np.ones(n), grid, model, nugget=0.1, transform_func=transform)
    x = np.random.randn(2) * 10
    grads = jacobian(reparamed_cov)(x, lags).T
    assert_allclose(grads, dw.d_cov_func(transform(x)))


def test_dewhittle_fisher():
    """test to confirm Arthur's H to my H, too much replication?"""

    from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
    from debiased_spatial_whittle.likelihood import DebiasedWhittle

    n = (64, 64)
    grid = RectangularGrid(n)
    model = ExponentialModel()
    params = abs(np.random.randn(2) * [3, 1])
    model.rho = params[0]
    model.sigma = params[1]
    model.nugget = 0.1

    dw = DeWhittle(
        np.ones(n), grid, ExponentialModel(), nugget=0.1, transform_func=None
    )

    p = Periodogram()
    ep = ExpectedPeriodogram(grid, p)
    d = DebiasedWhittle(p, ep)
    H = d.fisher(model, Parameters([model.rho, model.sigma]))
    assert np.all(np.diag(H) >= 0)

    H2 = dw.fisher(params)
    assert_allclose(H, H2)


if __name__ == "__main__":
    test_likelihood_dewhittle_grad()
    test_likelihood_whittle_grad()
    test_likelihood_gauss_grad()
    test_dewhittle_d_cov_func()
    test_dewhittle_fisher()
