import sys

import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    Parameters,
)
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram

n = (128, 128)
rho, sigma = 6, 3

grid = RectangularGrid(n)
model = SquaredExponentialModel()
model.rho = rho
model.sigma = sigma
model.nugget = 0.1

params = Parameters((model.rho, model.sigma))

per = Periodogram()
ep = ExpectedPeriodogram(grid, per)
db = DebiasedWhittle(per, ep)

sampler = SamplerOnRectangularGrid(model, grid)
sampler.n_sims = 300
z = sampler()

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(z, origin="lower", cmap="Spectral")
plt.show()

# expected hessian
hmat = db.fisher(model, params)
print(hmat)

# variance matrix of the score
jmat_mcmc = db.jmatrix(model, params, mcmc_mode=True)
jmat = db.jmatrix_sample(model, params, n_sims=1000)
print(jmat_mcmc)
print(jmat)

# variance of estimates
# cov_mat_mcmc = db.variance_of_estimates(model, model.params, jmat_mcmc)
cov_mat_sample = db.variance_of_estimates(model, params, jmat)

print("--------------")
# print(cov_mat_mcmc)
print(cov_mat_sample)

if (
    input("Run Monte Carlo simulations to compare with predicted variance (y/n)?")
    != "y"
):
    sys.exit(0)

import numpy as np

n_samples = int(input("Number of Monte Carlo samples?"))

estimates = np.zeros((n_samples, len(model.params)))
dw = Estimator(db)

for i in range(n_samples):
    print("------------")
    z = sampler()
    model_est = SquaredExponentialModel()
    model_est.nugget = model.nugget.value
    model_est.sigma.init_guess = 10
    dw(model_est, z)
    print(model_est)
    estimates[i, :] = model_est.params.values

print(np.cov(estimates.T))
print(cov_mat_sample)
