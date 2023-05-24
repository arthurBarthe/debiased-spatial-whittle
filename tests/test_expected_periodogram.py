import numpy as np
from numpy.testing import assert_allclose
from debiased_spatial_whittle import exp_cov, sim_circ_embedding, compute_ep_old, periodogram
from debiased_spatial_whittle.periodogram import autocov
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, SeparableExpectedPeriodogram, ExpectedPeriodogram
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, ExponentialModelUniDirectional, SeparableModel, Parameters
from debiased_spatial_whittle.confidence import CovarianceFFT

def test_non_negative():
    """
    This test verifies that the expected periodogram is non-negative.
    """
    m, n = 128, 64
    cov_func = lambda lags: exp_cov(lags, rho=10.)
    z = sim_circ_embedding(cov_func, (m, n))[0]
    e_per = compute_ep_old(cov_func, np.ones_like(z))
    assert np.all(e_per >= 0)

def test_autocov_1():
    """
    This test verifies that the returned covariances correspond to lags 0, 1, 2, -2, -1 when shape = (3,)
    :return:
    """
    cov_func = lambda x: x
    shape = (3, )
    acv = autocov(cov_func, shape)
    assert np.all(acv == [0., 1., 2., -2., -1.])


def test_autocov_2():
    """
    This test verifies that the returned covariances correspond to lags 0, 1, 2, -2, -1 when shape = (3,)
    :return:
    """
    cov_func = lambda x: x
    shape = (4, )
    acv = autocov(cov_func, shape)
    assert np.all(acv == [0., 1., 2., 3., -3., -2., -1.])


def test_compare_to_mean():
    m, n = 128, 64
    cov_func = lambda lags: exp_cov(lags, rho=15.)
    mean_per = np.zeros((m, n))
    grid = np.ones((m, n))
    n_samples = 50
    for i in range(n_samples):
        z, _ = sim_circ_embedding(cov_func, (m, n))
        per = periodogram(z, grid)
        mean_per = i / (i + 1) * mean_per + 1 / (i + 1) * per
    e_per = compute_ep_old(cov_func, grid)
    assert_allclose(mean_per, e_per, rtol=1)


def test_compare_to_mean2():
    m, n = 8, 8
    cov_func = lambda lags: exp_cov(lags, rho=2., sigma=2)
    mean_per = np.zeros((m, n))
    grid = np.ones((m, n))
    n_samples = 10000
    for i in range(n_samples):
        z, _ = sim_circ_embedding(cov_func, (m, n))
        per = periodogram(z, grid)
        mean_per = i / (i + 1) * mean_per + 1 / (i + 1) * per
    e_per = compute_ep_old(cov_func, grid)
    print(mean_per / e_per)
    assert_allclose(mean_per, e_per, rtol=0.1)


def test_separable_expected_periodogram():
    """
    This test verifies that for a separable model, the expected periodogram is the same when computed using separability
    and when not using it.
    :return:
    """
    rho_0 = 8
    m1 = ExponentialModelUniDirectional(axis=0)
    m1.rho = rho_0
    m1.sigma = 1
    m2 = ExponentialModelUniDirectional(axis=1)
    m2.rho = 32
    m2.sigma = 2
    model = SeparableModel((m1, m2))
    g = RectangularGrid((64, 64))
    p = Periodogram()
    ep1 = ExpectedPeriodogram(g, p)
    ep2 = SeparableExpectedPeriodogram(g, p)
    assert_allclose(ep1(model), ep2(model))


def test_periodogram_oop():
    """
    This test checks that the periodogram in the OOP implementation is the same as the one in the non-oop implementation
    :return:
    """
    g = RectangularGrid((64, 64))
    p_op = Periodogram()
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    sampler = SamplerOnRectangularGrid(model, g)
    z = sampler()
    p = p_op(z)
    p2 = periodogram(z, np.ones_like(z))
    assert_allclose(p, p2)


def test_expected_periodogram_oop():
    """
    This test checks that the expected periodogram in the OOP implementation is the same as the one in the non-oop
    implementation.
    :return:
    """
    g = RectangularGrid((64, 64))
    p = Periodogram()
    ep_op = ExpectedPeriodogram(g, p)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    ep_oop = ep_op(model)
    # old version
    cov_func = lambda x: exp_cov(x, 10)
    ep_old = compute_ep_old(cov_func, np.ones((64, 64)))
    assert_allclose(ep_old, ep_oop, rtol=1e-2)


def test_gradient_expected_periodogram():
    """
    This test verifies that the analytical gradient of the expected periodogram is close to a
    numerical approximation to that gradient
    :return:
    """
    g = RectangularGrid((8, 8))
    p = Periodogram()
    ep_op = ExpectedPeriodogram(g, p)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 4
    epsilon = 1e-6
    ep1 = ep_op(model)
    model.rho = model.rho.value + epsilon
    ep2 = ep_op(model)
    g = ep_op.gradient(model, Parameters([model.rho, ]))[:, :, 0]
    g2 = (ep2 - ep1) / epsilon
    assert_allclose(g, g2, rtol=1e-3)


def test_cov_dft_sum():
    """
    In this test we check that we get the same result by:
    1. Computing the covariance matrix of the covariance of the DFT, and summing its squared absolute value
    2. Computing the squared absolute values of the diagonals of the covariance of the DFT, and summing
    Returns
    -------

    """
    model = ExponentialModel()
    model.sigma = 2
    model.rho = 4
    n = (8, 8)
    g = RectangularGrid(n)
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    cov_mat = ep.cov_dft_matrix(model)
    s1 = np.sum(np.abs(cov_mat) ** 2)
    cov_fft = CovarianceFFT(g)
    s2 = cov_fft.exact_summation1(model, ep, normalize=False)
    print(s1, s2)
    assert_allclose(s1, s2)


def test_cov_dft_quad():
    """
    In this test we check that we get the same result by:
    1. Computing the covariance matrix of the covariance of the DFT, and summing its squared absolute value
    2. Computing the squared absolute values of the diagonals of the covariance of the DFT, and summing
    Returns
    -------

    """
    model = ExponentialModel()
    model.sigma = 2
    model.rho = 4
    n = (8, 8)
    g = RectangularGrid(n)
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    f = np.random.randn(*n)
    f2 = np.random.randn(*n)
    cov_mat = ep.cov_dft_matrix(model).reshape(n[0] * n[1], n[0] * n[1])
    cov_mat = np.abs(cov_mat)**2
    s1 = np.dot(f.reshape((1, -1)), np.dot(cov_mat, f2.reshape((-1, 1))))
    cov_fft = CovarianceFFT(g)
    s2 = cov_fft.exact_summation1(model, ep, f=f, f2=f2, normalize=False)
    print(s1, s2)
    assert_allclose(s1, s2)


def test_rel_dft():
    """
    In this test we check that we get the same result by:
    1. Computing the covariance matrix of the relation of the DFT, and summing its squared absolute value.
    2. Computing the squared absolute values of the diagonals of the relation of the DFT, and summing
    Returns
    -------

    """
    model = ExponentialModel()
    model.sigma = 2
    model.rho = 4
    n = (8, 8)
    g = RectangularGrid(n)
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    cov_mat = ep.rel_dft_matrix(model)
    s1 = np.sum(np.abs(cov_mat) ** 2)
    cov_fft = CovarianceFFT(g)
    s2 = cov_fft.exact_summation2(model, ep, normalize=False)
    print(s1, s2)
    assert_allclose(s1, s2)

def test_rel_dft_quad():
    """
    In this test we check that we get the same result by:
    1. Computing the covariance matrix of the covariance of the DFT, and summing its squared absolute value
    2. Computing the squared absolute values of the diagonals of the covariance of the DFT, and summing
    Returns
    -------

    """
    model = ExponentialModel()
    model.sigma = 2
    model.rho = 4
    n = (8, 8)
    g = RectangularGrid(n)
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    f = np.random.randn(*n)
    f2 = np.random.randn(*n)
    cov_mat = ep.rel_dft_matrix(model).reshape(n[0] * n[1], n[0] * n[1])
    cov_mat = np.abs(cov_mat)**2
    s1 = np.dot(f.reshape((1, -1)), np.dot(cov_mat, f2.reshape((-1, 1))))
    cov_fft = CovarianceFFT(g)
    s2 = cov_fft.exact_summation2(model, ep, f=f, f2=f2, normalize=False)
    print(s1, s2)
    assert_allclose(s1, s2)