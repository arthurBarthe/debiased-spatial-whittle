import numpy as np
from debiasedwhittle import exp_cov, sim_circ_embedding, compute_ep, periodogram
from debiased_spatial_whittle.expected_periodogram import autocov

def test_non_negative():
    """
    This test verifies that the expected periodogram is non-negative.
    """
    m, n = 128, 64
    cov_func = lambda lags: exp_cov(lags, rho=10.)
    z = sim_circ_embedding(cov_func, (m, n))
    e_per = compute_ep(cov_func, np.ones_like(z))
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
    e_per = compute_ep(cov_func, grid)
    assert np.all(abs(mean_per - e_per))