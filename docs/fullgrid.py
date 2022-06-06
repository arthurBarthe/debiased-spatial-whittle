import numpy as np
import matplotlib.pyplot as plt

from debiasedwhittle import sim_circ_embedding, sq_exp_cov, exp_cov, exp_cov, fit, matern15_cov_func


shape = (512 * 1, 512 * 1)
cov_func = lambda lags: sq_exp_cov(lags, rho=32.)
z = sim_circ_embedding(cov_func, shape)[0]
plt.figure()
plt.imshow(z, cmap='BrBG')
plt.show()

est3 = fit(z, np.ones_like(z), sq_exp_cov, [1, ], fold=True, taper=True)
est2 = fit(z, np.ones_like(z), sq_exp_cov, [1, ], fold=True)
print(est3)
print(est2)