import numpy as np
from debiasedwhittle import expected_periodogram, exp_cov, sim_circ_embedding, compute_ep, periodogram

def test_non_negative():
    m, n = 128, 64
    cov_func = lambda lags: exp_cov(lags, rho=10.)
    z = sim_circ_embedding(cov_func, (m, n))
    e_per = compute_ep(cov_func, np.ones_like(z))
    assert np.all(e_per > 0)

def test_compare_to_mean():
    m, n = 128, 64
    cov_func = lambda lags: exp_cov(lags, rhp=15.)
    mean_per = np.zeros((m, n))
    grid = np.ones((m, n))
    n_samples = 10
    for i in range(n_samples):
        z = sim_circ_embedding(cov_func, (m, n))
        per = periodogram(z, grid)
        mean_per = i / (i + 1) * mean_per + 1 / (i + 1) * per
    e_per = compute_ep(cov_func, grid)
    assert np.all(abs(mean_per - e_per))