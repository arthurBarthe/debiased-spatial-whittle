import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from debiased_spatial_whittle import sim_circ_embedding, sq_exp_cov, exp_cov, exp_cov, fit


shape = (512 * 1, 512 * 1)
x_0, y_0, diameter = 256, 256, 512
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
circle = ((x - x_0)**2 + (y - y_0)**2) <= 1 / 4 * diameter**2
circle = circle * 1.
cov_func = lambda lags: sq_exp_cov(lags, rho=15.)
z = sim_circ_embedding(cov_func, shape)[0]
z *= circle
plt.figure()
plt.imshow(z, cmap='Spectral')
plt.show()

est = fit(z, circle, sq_exp_cov, [1., ], fold=False)
print(est)