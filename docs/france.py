import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from debiasedwhittle import sim_circ_embedding, sq_exp_cov, exp_cov, exp_cov, fit
import debiasedwhittle.grids as grids

cov = sq_exp_cov
shape = (620 * 1, 620 * 1)
cov_func = lambda lags: cov(lags, rho=32.)
img = grids.ImgGrid(shape).get_new()
z = sim_circ_embedding(cov_func, shape)[0] * img
est = fit(z, img, cov, [1., ])
print(est)

plt.imshow(z, origin='lower', cmap='Spectral')
plt.show()