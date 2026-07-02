from debiased_spatial_whittle.backend import BackendManager
np = BackendManager.get_backend()

from numpy.testing import assert_allclose

from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.periodogram import (
    Periodogram,
    ExpectedPeriodogram,
    compute_ep_old,
)
from debiased_spatial_whittle.inference.multivariate_periodogram import (
    Periodogram as PeriodogramMulti,
)
from debiased_spatial_whittle.inference.likelihood import (
    DebiasedWhittle,
    Estimator,
    MultivariateDebiasedWhittle,
)
from debiased_spatial_whittle.inference.old import whittle, periodogram
from debiased_spatial_whittle.sampling.simulation import (
    SamplerOnRectangularGrid,
    SamplerBUCOnRectangularGrid,
)
from debiased_spatial_whittle.models.univariate import (
    ExponentialModel,
    SquaredExponentialModel,
)
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.models.old import exp_cov


def test_oop():
    """
    This test verifies that the oop implementation gives the same debiased whittle likelihood as the old
    implementation.
    :return:
    """
    rho = 10
    rho_lkh = 15
    g = RectangularGrid((256, 256))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel(rho=rho, sigma=1)
    sampler = SamplerOnRectangularGrid(model, g)
    z = sampler()
    model.rho = rho_lkh
    lkh_oop = d(z, model)
    # old version
    g = np.ones((256, 256))
    cov_func = lambda x: exp_cov(x, rho_lkh)
    e_per = compute_ep_old(cov_func, g)
    lkh_old = whittle(periodogram(z, g), e_per)
    assert lkh_old == lkh_oop


def test_model_array():
    """
    In this test we compute the debiased whittle for several model parameter values in a vectorized fashion.
    """
    rho = 10
    g = RectangularGrid((128, 128))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = rho
    sampler = SamplerOnRectangularGrid(model, g)
    z = sampler()
    model.rho = np.arange(1, 20)
    lkh = d(z, model)
    print(lkh)
    assert lkh.shape == (19,)
    assert lkh[9] < lkh[18]


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
    z = sampler()
    
    # Get parameter name for rho
    param_names = [model.parameter_names[0]]
    
    # Compute likelihood and gradient using the gradient method
    lkh = d(z, model)
    grad_dict = d.gradient(z, model, param_names=param_names)
    grad = grad_dict[param_names[0]]
    
    # Compute numerical gradient
    epsilon = 1e-6
    model.rho = model.rho + epsilon
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
    model = SquaredExponentialModel(rho=3, sigma=1)
    bvm = BivariateUniformCorrelation(model)
    bvm.r = 0.3
    bvm.f = 1.5
    sampler = SamplerBUCOnRectangularGrid(bvm, g)
    z = sampler()
    dbw = MultivariateDebiasedWhittle(p, ep_op)
    epsilon = 1e-8
    
    # Get parameter name for r (first parameter of BivariateUniformCorrelation)
    param_name = bvm.parameter_names[0]
    param_names = [param_name]
    
    # Compute likelihood and gradient using the gradient method
    lkh = dbw(z, bvm)
    grad_dict = dbw.gradient(z, bvm, param_names=param_names)
    grad = grad_dict[param_name]
    
    # Compute numerical gradient
    old_value = getattr(bvm, 'r')
    new_value = old_value + epsilon
    setattr(bvm, 'r', new_value)
    lkh2 = dbw(z, bvm)
    grad_num = (lkh2 - lkh) / epsilon
    assert_allclose(grad, grad_num, rtol=0.001)
    setattr(bvm, 'r', old_value)


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
    param_names = [model.parameter_names[0]]
    h = d.fisher(model, param_names=param_names)
    print(h)
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
    bvm.r = 0.3
    bvm.f = 1.5
    sampler = SamplerBUCOnRectangularGrid(bvm, g)
    dbw = MultivariateDebiasedWhittle(p, ep_op)
    
    # Get parameter names for r and f
    param_names = [bvm.parameter_names[0], bvm.parameter_names[1]]
    
    h = dbw.fisher(bvm, param_names=param_names)
    assert np.all(np.diag(h) > 0)


def test_jmat():
    """
    Compares the predicted covariance matrix of the score with the sample variance of the score
    obtained from Monte Carlo simulations
    """
    g = RectangularGrid((16, 16))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel(rho=2, sigma=1)
    sampler = SamplerOnRectangularGrid(model, g)
    params = [model.param.rho, model.param.sigma]
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
    covmat = e.covmat(model, [model.param.rho, model.param.sigma])
    print(covmat)
    assert np.all(np.diag(covmat) >= 0)


def test_jmatrix_sample():
    g = RectangularGrid((256, 256))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel(rho=2, sigma=1)
    param_names = [model.parameter_names[0], model.parameter_names[1]]
    jmat = d.jmatrix_sample(model, param_names=param_names)
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
    bvm.r = 0.3
    bvm.f = 1.5
    dbw = MultivariateDebiasedWhittle(p, ep_op)
    param_names = [bvm.parameter_names[0], bvm.parameter_names[1]]
    jmat = dbw.jmatrix_sample(bvm, param_names=param_names)
    assert jmat.shape == (2, 2)
    assert np.all(np.diag(jmat) > 0)
