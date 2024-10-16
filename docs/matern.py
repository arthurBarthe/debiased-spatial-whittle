"""In this example we estimate the three parameters of a matern covariance function. This is quite slow due
to the need to evaluate the modified Bessel function of the second order. """

import numpy as np
from debiased_spatial_whittle import sim_circ_embedding, fit, matern
import matplotlib.pyplot as plt

rho, nu, sigma = 8, 1.2, 1.
init_guess = np.array([10., 0.5, 1.1])

cov = matern
cov_func = lambda lags: cov(lags, rho, nu, sigma)

shape = (256, 256)
z, _ = sim_circ_embedding(cov_func, shape)
plt.imshow(z, cmap='coolwarm')
plt.show()
est = fit(z, np.ones_like(z), cov, init_guess, fold=True)
print(est)


def run_experiment(sim_params, est_params, n_samples=2):
    est = np.zeros((n_samples, 3))
    for i in range(n_samples):
        print(i, end=': ')
        z, _ = sim_circ_embedding(*sim_params)
        est[i, :] = fit(z, np.ones_like(z), *est_params)
        print(est[i])
    return est


sim_params = (cov_func, shape)
est_params = (cov, init_guess, True)
estimates = run_experiment(sim_params, est_params)
print(np.mean(estimates, axis=0))
