import numpy as np
import matplotlib.pyplot as plt

from debiasedwhittle import sim_circ_embedding, sq_exp_cov, exp_cov, exp_cov, fit, matern15_cov_func
import debiasedwhittle.grids as grids


cov = sq_exp_cov
init_guess = np.array([1., ])




# probability of each point
p_obs = 0.5

shape = (620 * 1, 620 * 1)

# make map from france image
img = grids.ImgGrid(shape).get_new()

cov_func = lambda lags: cov(lags, rho=15.)
z = sim_circ_embedding(cov_func, shape)[0]
est = fit(z, np.ones_like(z), cov, init_guess, fold=False)
print(est)

g = (np.random.rand(*shape) <= p_obs) * img
z2 = z * g
est = fit(z2, g, cov, init_guess, fold=False)
print(est)

plt.imshow(z2, cmap='coolwarm')
plt.show()