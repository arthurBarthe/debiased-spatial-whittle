"""In this example we estimate the three parameters of a matern covariance function. This is quite slow due
to the need to evaluate the modified Bessel function of the second order. """

import numpy as np
from debiasedwhittle import sim_circ_embedding, fit, matern
import matplotlib.pyplot as plt

rho, nu, sigma = 15, 0.3, 1.
init_guess = np.array([10., 0.5, 1.1])

cov = matern
cov_func = lambda lags: cov(lags, rho, nu, sigma)

shape = (512, 512)
z, _ = sim_circ_embedding(cov_func, shape)
plt.imshow(z, cmap='coolwarm')
plt.show()
est = fit(z, np.ones_like(z), cov, init_guess, fold=True)
print(est)


def run_experiment(sim_params, est_params, n_samples=10):
    est = np.zeros(n_samples)
    for i in range(n_samples):
        print(i, end=': ')
        z, _ = sim_circ_embedding(*sim_params)
        est[i] = fit(z, np.ones_like(z), *est_params)[0]
        print(est[i])
    return est


sim_params = (cov_func, shape)
est_params = (cov, init_guess, True)
estimates = run_experiment(sim_params, est_params)
