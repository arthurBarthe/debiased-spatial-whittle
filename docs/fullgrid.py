import numpy as np
import matplotlib.pyplot as plt

from debiasedwhittle import sim_circ_embedding, sq_exp_cov, exp_cov, exp_cov, fit, matern15_cov_func


def exp_cov1d(x: np.ndarray, rho: float):
    return np.exp(-np.abs(x) / rho) + 0.00 * (x == np.zeros_like(x))

def exp_cov_separable(lags, rho, sigma=1.):
    return exp_cov1d(lags[0], rho) * exp_cov1d(lags[1], rho)

shape = (64, 64)
n = shape
cov_func_family = exp_cov_separable
rho = 8

# grid
g = np.ones(n)

g_circle = np.ones(n)
xs = np.meshgrid(np.arange(n[0]), np.arange(n[1]))
d = np.sqrt((xs[0] - n[0] // 2)**2 + (xs[1] - n[1] // 2)**2)
g_circle[d > n[0] / 3] = 0

g = g_circle

cov_func = lambda lags: cov_func_family(lags, rho=rho)
z = sim_circ_embedding(cov_func, shape)[0]
z *= g
plt.figure()
plt.imshow(z, cmap='BrBG')
plt.show()

ests = []
for i in range(1000):
    z = sim_circ_embedding(cov_func, shape)[0]
    z *= g
    est2 = fit(z, g, cov_func_family, [1, ], fold=True)
    print(est2)
    ests.append(est2)

print(np.std(ests))