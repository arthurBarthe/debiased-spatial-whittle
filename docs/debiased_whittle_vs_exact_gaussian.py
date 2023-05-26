import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternCovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.likelihood import DebiasedWhittle
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.bayes.likelihoods import Gaussian, Whittle

m = 32
nu = 2.5

model_family = MaternCovarianceModel

grid = RectangularGrid((m, m))
model = model_family()
model.rho = 4
model.sigma = 1
model.nugget = 0.2
model.nu = nu

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

model_eval = model_family()
model_eval.sigma = 1
model_eval.nugget = 0.2
model_eval.nu = nu

p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
db = DebiasedWhittle(p, ep)

thetas = np.arange(1, 6, 0.1)
e_values = np.zeros_like(thetas)
r_values = np.zeros_like(thetas)
r_values_gaussian = np.zeros_like(thetas)
r_values_whittle = np.zeros_like(thetas)

if m <= 64:
    gaussian_likelihood = Gaussian(z, grid, model_eval, 0.2)
whittle_likelihood = Whittle(z, grid, model_eval, 0.2)

for i, theta in enumerate(thetas):
    print(i)
    model_eval.rho = theta
    e_values[i] = db.expected(model, model_eval) / 2 + m**2 / 2 * np.log(2 * np.pi)
    r_values[i] = db(z, model_eval) * m**2 / 2 + m**2 / 2 * np.log(2 * np.pi)
    model_eval.rho = None
    if m <= 64:
        r_values_gaussian[i] = -gaussian_likelihood([np.log(theta), ])
    #r_values_whittle[i] = -whittle_likelihood([np.log(theta)]) + m**2 / 2 * np.log(2 * np.pi)

plt.figure()
plt.plot(thetas, e_values)
plt.plot(thetas, r_values)
plt.plot(thetas, r_values_whittle)
plt.plot(thetas, r_values_gaussian)
plt.show()