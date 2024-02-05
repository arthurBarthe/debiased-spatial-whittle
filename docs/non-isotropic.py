"""In this example, we use a non-isotropic exponential covariance function, with two length scales and an angle
theta. We jointly estimate the three parameters from the simulated data."""

import numpy as np
from debiased_spatial_whittle import exp_cov_anisotropic, sim_circ_embedding, fit
import matplotlib.pyplot as plt

rho_1, rho_2, theta = 30, 5, 0.8
init_guess = np.array([1., 1., 0.5])

cov = exp_cov_anisotropic
cov_func = lambda lags: cov(lags, rho_1, rho_2, theta)

shape = (256, 256)
z = sim_circ_embedding(cov_func, shape)[0]
est = fit(z, np.ones_like(z), cov, init_guess)
est[-1] %= np.pi
print(est)
plt.imshow(z, cmap='Spectral')
plt.show()