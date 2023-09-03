import sys

import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram


n = (256, 256)
rho, sigma = 3, 1

grid = RectangularGrid(n)
model = ExponentialModel()
model.rho = rho
model.sigma = sigma

per = Periodogram()
ep = ExpectedPeriodogram(grid, per)
db = DebiasedWhittle(per, ep)

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(z, origin='lower', cmap='Spectral')
plt.show()

# Fisher matrix
hmat = db.fisher(model, model.params)
print(hmat)

# variance matrix of the score
#jmat = db.jmatrix(model, model.params)
jmat_sample = db.jmatrix_sample(model, model.params, n_sims=500)
#print(jmat)
print(jmat_sample)

# variance of estimates
#cov_mat = db.variance_of_estimates(model, model.params, jmat)
cov_mat_sample = db.variance_of_estimates(model, model.params, jmat_sample)

print('--------------')
#print(cov_mat)
print(cov_mat_sample)

if input('Run Monte Carlo simulations to compare with predicted variance (y/n)?') != 'y':
    sys.exit(0)

import numpy as np
n_samples = int(input('Number of Monte Carlo samples?'))

estimates = np.zeros((n_samples, len(model.params)))
dw = Estimator(db)

for i in range(n_samples):
    print('------------')
    z = sampler()
    model_est = ExponentialModel()
    dw(model_est, z)
    print(model_est.params)
    estimates[i, :] = model_est.params.values

print(np.cov(estimates.T))
print(cov_mat_sample)