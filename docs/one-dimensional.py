"""
A 1-d example of the Debiased Whittle.
"""

import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle import sim_circ_embedding, exp_cov, fit

shape = (2**14 * 1,)
cov_func_family = exp_cov
cov_func = lambda lags: cov_func_family(lags, rho=10.0)
z = sim_circ_embedding(cov_func, shape)[0].reshape((-1, 1))
plt.figure()
plt.plot(z)
plt.show()

est = fit(
    z,
    np.ones_like(z),
    cov_func_family,
    [
        1,
    ],
    fold=True,
)
print(est)
