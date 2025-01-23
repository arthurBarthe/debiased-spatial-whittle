import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import slogdet

from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    MaternCovarianceModel,
)
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.likelihood import DebiasedWhittle
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.bayes.likelihoods import Gaussian, Whittle

m = 16
nu = 0.1

model_family = MaternCovarianceModel

grid = RectangularGrid((m, m))
model = model_family()
model.rho = 5
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


def expected_gaussian_likelihood(true_model, est_model, grid: RectangularGrid):
    from scipy.linalg import inv

    lags = grid.lag_matrix
    cov_mat = true_model(lags)
    cov_mat_est = est_model(lags)
    n = grid.n_points
    term1 = -n / 2 * np.log(2 * np.pi)
    term2 = -1 / 2 * np.trace(np.dot(inv(cov_mat_est), cov_mat))
    _, term3 = slogdet(cov_mat_est)
    return term1 + term2 - term3 / 2


thetas = np.linspace(model.rho.value / 2, model.rho.value * 2, 10, dtype=np.float64)
e_values = np.zeros_like(thetas)
r_values = np.zeros_like(thetas)
r_values_gaussian = np.zeros_like(thetas)
r_values_egaussian = np.zeros_like(thetas)
r_values_whittle = np.zeros_like(thetas)

if m <= 64:
    gaussian_likelihood = Gaussian(z, grid, model_eval, 0.2)
whittle_likelihood = Whittle(z, grid, model_eval, 0.2)

for i, theta in enumerate(thetas):
    print(i)
    model_eval.rho = theta
    e_values[i] = db.expected(model, model_eval) / 2 + m**2 / 2 * np.log(2 * np.pi)
    r_values[i] = db(z, model_eval) * m**2 / 2 + m**2 / 2 * np.log(2 * np.pi)
    if m <= 64:
        r_values_gaussian[i] = -gaussian_likelihood(
            [
                np.log(theta),
            ]
        )
        r_values_egaussian[i] = -expected_gaussian_likelihood(model, model_eval, grid)
    # r_values_whittle[i] = -whittle_likelihood([np.log(theta)]) + m**2 / 2 * np.log(2 * np.pi)

plt.figure()
plt.plot(thetas, e_values / m**2, "-*", label="Expected DBW")
plt.plot(thetas, r_values / m**2, label="DBW")
# plt.plot(thetas, r_values_whittle, label='Whittle')
plt.plot(thetas, r_values_gaussian / m**2, label="Gaussian")
plt.plot(thetas, r_values_egaussian / m**2, label="Expected Gaussian")
plt.legend()
plt.show()
