import numpy as np
from numpy.testing import assert_allclose

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import (
    Periodogram,
    ExpectedPeriodogram,
    compute_ep_old,
)
from debiased_spatial_whittle.multivariate_periodogram import (
    Periodogram as PeriodogramMulti,
)
from debiased_spatial_whittle.likelihood import (
    DebiasedWhittle,
    whittle,
    Estimator,
    periodogram,
    MultivariateDebiasedWhittle,
)
from debiased_spatial_whittle.simulation import (
    SamplerOnRectangularGrid,
    SamplerBUCOnRectangularGrid,
)
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    Parameters,
    BivariateUniformCorrelation,
)
from debiased_spatial_whittle.cov_funcs import exp_cov


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
    e_per = compute_ep_old(cov_func, g)
    lkh_old = whittle(periodogram(z, g), e_per)
    assert lkh_old == lkh_oop


def test_whittle_grad():
    """
    This tests the implementation of the gradient of the Whittle likelihood
    Returns
    -------
    """
    g = RectangularGrid((8, 8))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 4
    sampler = SamplerOnRectangularGrid(model, g)
    p = Parameters(
        [
            model.rho,
        ]
    )
    z = sampler()
    lkh, grad = d(z, model, params_for_gradient=p)
    epsilon = 1e-6
    model.rho = model.rho.value + epsilon
    lkh2 = d(z, model)
    grad_num = (lkh2 - lkh) / epsilon
    assert_allclose(grad, grad_num, rtol=0.001)


def test_whittle_grad_multi():
    """
    Tests the implementation of the gradient of the whittle likelihood in the multivariate case
    """
    g = RectangularGrid((32, 32), nvars=2)
    p = PeriodogramMulti()
    ep_op = ExpectedPeriodogram(g, p)
    model = SquaredExponentialModel()
    model.rho = 3
    model.sigma = 1
    model.nugget = 0.2
    bvm = BivariateUniformCorrelation(model)
    bvm.r_0 = 0.3
    bvm.f_0 = 1.5
    sampler = SamplerBUCOnRectangularGrid(bvm, g)
    z = sampler()
    dbw = MultivariateDebiasedWhittle(p, ep_op)
    epsilon = 1e-8
    lkh, grad = dbw(z, bvm, bvm.params)
    for i, p in enumerate(bvm.params):
        print(p)
        p.value = p.value + epsilon
        lkh2 = dbw(z, bvm)
        grad_num = (lkh2 - lkh) / epsilon
        assert_allclose(grad[i], grad_num, rtol=0.001)
        p.value = p.value - epsilon


def test_hessian_diagonal():
    """
    Basic test for the hessian that verifies that the hessian has non-negative terms on the diagonal
    and that the hessian has the appropriate shape
    """
    rho = 10
    g = RectangularGrid((32, 32))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = rho
    h = d.fisher(
        model,
        Parameters(
            [
                model.rho,
            ]
        ),
    )
    print(h)
    # assert h.shape == (2, 2)
    assert np.all(np.diag(h) >= 0)


def test_fisher_multivariate():
    """
    Runs the fisher method in the multivariate case. Checks that the diagonal of the result is positive.
    """
    g = RectangularGrid((32, 32), nvars=2)
    p = PeriodogramMulti()
    ep_op = ExpectedPeriodogram(g, p)
    model = SquaredExponentialModel()
    model.rho = 3
    model.sigma = 1
    model.nugget = 0.2
    bvm = BivariateUniformCorrelation(model)
    bvm.r_0 = 0.3
    bvm.f_0 = 1.5
    sampler = SamplerBUCOnRectangularGrid(bvm, g)
    z = sampler()
    dbw = MultivariateDebiasedWhittle(p, ep_op)
    h = dbw.fisher(bvm, bvm.params)
    assert np.all(np.diag(h) > 0)


def test_jmat():
    """
    Compares the predicted covariance matrix of the score with the sample variance of the score
    obtained from Monte Carlo simulations

    Not passing rn. Different possibilities:
    1. The covariances of the periodogram are not right. However, a basic check of that (summation) passes.
    We could check pointwise.
    2. The normalization is not right. Does not appear to be the case at first sight...
    3. The derivatives of the expected periodogram are not right
    4. The indexing in the summation is not right.
    5. The gradient of the likelihood is not right.
    """
    g = RectangularGrid((16, 16))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 2
    sampler = SamplerOnRectangularGrid(model, g)
    params = model.params
    print(params)
    jmat = d.jmatrix(model, params)
    n_samples = 1000
    estimates = []
    for i in range(n_samples):
        z = sampler()
        lkh, grad = d(z, model, params_for_gradient=params)
        estimates.append(grad)
    estimates = np.array(estimates)
    sample_cov_mat = np.cov(estimates.T)
    # sample_cov_mat = 1 / n_samples * np.dot(estimates.T, estimates)
    print(jmat)
    print(sample_cov_mat)
    assert_allclose(
        jmat,
        sample_cov_mat,
        0.15,
    )


def test_covmat():
    """
    Test for the approximation of the covariance matrix of the debiased whittle estimates.
    """
    g = RectangularGrid((32, 32))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    e = Estimator(d)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 2
    covmat = e.covmat(model, model.params)
    print(covmat)
    assert np.all(np.diag(covmat) >= 0)


def test_jmatrix_sample():
    g = RectangularGrid((256, 256))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 2
    jmat = d.jmatrix_sample(model, model.params)
    print(jmat)


def test_jmatrix_sample_multivariate():
    g = RectangularGrid((32, 32), nvars=2)
    p = PeriodogramMulti()
    ep_op = ExpectedPeriodogram(g, p)
    model = SquaredExponentialModel()
    model.rho = 3
    model.sigma = 1
    model.nugget = 0.2
    bvm = BivariateUniformCorrelation(model)
    bvm.r_0 = 0.3
    bvm.f_0 = 1.5
    dbw = MultivariateDebiasedWhittle(p, ep_op)
    jmat = dbw.jmatrix_sample(bvm, bvm.params)
    assert jmat.shape == (5, 5)
    assert np.all(np.diag(jmat) > 0)
