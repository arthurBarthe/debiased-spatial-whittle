import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from debiased_spatial_whittle import sim_circ_embedding, sq_exp_cov, exp_cov, exp_cov, fit


shape = (512 * 1, 512 * 1)
p_obs = 0.1
cov_func = lambda lags: exp_cov(lags, rho=15.)
z = sim_circ_embedding(cov_func, shape)[0]
g = (np.random.rand(*shape) <= p_obs) * 1.
z2 = z * g
est = fit(z2, g, exp_cov, [1., ])
print(est)

plt.imshow(z2)
plt.show()