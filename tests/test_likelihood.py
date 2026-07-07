from debiased_spatial_whittle.backend import BackendManager
np = BackendManager.get_backend()

assert_allclose = BackendManager.get_assert_allclose()

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
    MultivariateSamplerOnRectangularGrid,
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
    lkh_old = whittle(periodogram(z, g), e_per).item()
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
    assert_allclose(grad, grad_num, rtol=0.001, atol=1e-2)


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
    sampler = MultivariateSamplerOnRectangularGrid(bvm, g, p=2)
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
    assert_allclose(grad, grad_num, rtol=0.001, atol=0)
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
    param_names = [model.parameter_names[0], model.parameter_names[1]]
    print(param_names)
    jmat = d.jmatrix(model, param_names=param_names)
    jmat_sample = d.jmatrix_sample(model, param_names=param_names, n_sims=1000)
    print(jmat)
    print(jmat_sample)
    assert_allclose(
        jmat,
        jmat_sample,
        rtol=0.15,
        atol=0.1
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
    param_names = [model.parameter_names[0], model.parameter_names[1]]
    covmat = e.covmat(model, param_names=param_names)
    print(covmat)
    assert np.all(np.diag(covmat) >= 0)


def test_jmatrix_sample():
    g = RectangularGrid((32, 32))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel(rho=2, sigma=1)
    param_names = [model.parameter_names[0], model.parameter_names[1]]
    jmat = d.jmatrix_sample(model, param_names=param_names, n_sims=10)
    print(jmat)


def test_jmatrix_sample_multivariate():
    g = RectangularGrid((32, 32), nvars=2)
    p = PeriodogramMulti()
    ep_op = ExpectedPeriodogram(g, p)
    model = ExponentialModel()
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


def test_variance_of_estimates_sum_model():
    """
    Test that variance_of_estimates works with a sum of two SquaredExponentialModel instances.
    """
    from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
    
    # Create two SquaredExponentialModel instances with distinct names
    model1 = SquaredExponentialModel(rho=10, sigma=0.8, name="se1")
    model2 = SquaredExponentialModel(rho=5, sigma=0.5, name="se2")
    
    # Sum the two models
    model = model1 + model2
    print(model)
    
    grid = RectangularGrid((32, 32))
    
    # Create DebiasedWhittle
    periodogram = Periodogram()
    ep = ExpectedPeriodogram(grid, periodogram)
    dbw = DebiasedWhittle(periodogram, ep)
    
    # Get covariance matrix of estimates
    cov_mat = dbw.variance_of_estimates(model)
    print(cov_mat)
    
    # Check that the covariance matrix has the correct shape
    assert cov_mat.shape[0] == model.n_parameters
    assert cov_mat.shape[1] == model.n_parameters
    
    # Check that the covariance matrix is symmetric (within numerical tolerance)
    assert_allclose(cov_mat, cov_mat.T, rtol=1e-10, atol=1e-2)
