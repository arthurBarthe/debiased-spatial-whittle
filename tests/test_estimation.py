import numpy as np
from numpy.testing import assert_almost_equal
from debiased_spatial_whittle import exp_cov, sim_circ_embedding, compute_ep_old, periodogram
from debiased_spatial_whittle.periodogram import autocov
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, SeparableExpectedPeriodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator, fit, whittle
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, ExponentialModelUniDirectional, SeparableModel


def test_expcov():
    """
    This test draws a sample from an exponential covariance model, estimates the range and verifies
    that it is close to the true value.
    :return:
    """
    g = RectangularGrid((128, 128))
    rhos = [5, 10, 15, 20]
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    e = Estimator(d)
    model = ExponentialModel()
    model.sigma = 1
    for rho in rhos:
        model.rho = rho
        sampler = SamplerOnRectangularGrid(model, g)
        model_est = ExponentialModel()
        model_est.sigma = 1
        z = sampler()
        e(model_est, z)
        est_rho = model_est.rho
        print(est_rho.value)
        assert(abs(est_rho.value - rho) / rho < 0.05)


def test_separable_expcov():
    """
    This test checks estimation for a separable exponential covariance model
    :return:
    """
    rho_0 = 8
    m1 = ExponentialModel()
    m1.rho = rho_0
    m1.sigma = 1
    m2 = ExponentialModel()
    m2.rho = 32
    m2.sigma = 2
    model = SeparableModel((m1, m2), dims=[(0, ), (1, )])
    # simulation
    g = RectangularGrid((128, 128))
    sampler = SamplerOnRectangularGrid(model, g)
    z = sampler()
    # estimation
    model_est = model
    m1.rho = None
    m2.rho = None
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    #ep = SeparableExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    e = Estimator(d)
    e(model_est, z)
    rho_0_est = m1.rho.value
    rho_1_est = m2.rho.value
    assert np.abs(rho_0_est - rho_0) < 2


def test_oop_vs_old():
    """
    This test verifies that the oop implementation gives the same estimates as the non-oop version.
    :return:
    """
    # oop version
    g = RectangularGrid((128, 128))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    e = Estimator(d)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    sampler = SamplerOnRectangularGrid(model, g)
    model_est = ExponentialModel()
    model_est.sigma = 1
    z = sampler()
    e(model_est, z, opt_callback=lambda x: print('current oop: ', x))
    est_rho = model_est.rho.value
    # old version
    g = np.ones((128, 128))
    cov_func = exp_cov
    est_rho2 = fit(z, g, cov_func, [model_est.rho.init_guess], opt_callback=lambda x: print('current old: ', x))
    assert_almost_equal(est_rho, est_rho2[0])


def test_optim_with_gradient():
    """
    This test checks that the estimation procedures works correctly when using the gradient of the likelihood
    for the optimization.
    :return:
    """
    g = RectangularGrid((128, 128))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    e = Estimator(d, use_gradients=True)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    sampler = SamplerOnRectangularGrid(model, g)
    model_est = ExponentialModel()
    model_est.sigma = 1
    z = sampler()
    e(model_est, z, opt_callback=lambda x: print('current oop: ', x))
    est_rho = model_est.rho.value
    assert (abs(est_rho - 10) < 2)


def test_optim_with_gradient_shared_param():
    """
    This test checks that the estimation procedures works correctly when using the gradient of the likelihood
    for the optimization, in the case where a single parameter is used for two values in the model.
    :return:
    """
    rho_0 = 8
    m1 = ExponentialModel()
    m1.sigma = 1
    m2 = ExponentialModel()
    m2.sigma = 2
    model = SeparableModel((m1, m2), dims=[(0, ), (1, )])
    model.merge_parameters(('rho_0', 'rho_1'))
    m1.rho = rho_0
    # simulation
    g = RectangularGrid((128, 128))
    sampler = SamplerOnRectangularGrid(model, g)
    z = sampler()
    # estimation
    model_est = model
    m1.rho = None
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    # ep = SeparableExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    e = Estimator(d, use_gradients=True)
    e(model_est, z)
    rho_0_est = m1.rho.value
    rho_1_est = m2.rho.value
    print(rho_0_est, rho_1_est)
    assert np.abs(rho_0_est - rho_0) < 2