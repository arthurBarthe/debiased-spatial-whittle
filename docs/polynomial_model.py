import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import PolynomialModel, MaternCovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import PolynomialSamplerOnRectangularGrid

from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator


m, n = 512, 512

grid = RectangularGrid((m, n))

latent_model = MaternCovarianceModel()
latent_model.sigma = 1
latent_model.rho = 5
latent_model.nu = 2.5

model = PolynomialModel(latent_model)
model.a_1 = 0.05
model.b_1 = 0.8

sampler = PolynomialSamplerOnRectangularGrid(model, grid)
z = sampler()

plt.figure()
plt.imshow(z, cmap="Spectral")
plt.colorbar()
plt.show()

# compute periodogram and show residuals
periodogram = Periodogram()
per = periodogram(z)

expected_periodogram = ExpectedPeriodogram(grid, periodogram)
ep = expected_periodogram(model)

import numpy as np
from numpy.fft import fftshift

plt.figure()
plt.imshow(fftshift((per / ep)))
plt.colorbar()
plt.show()

print(np.mean(per / ep))

# estimation
sdw = DebiasedWhittle(periodogram, expected_periodogram)
print(f"Likelihood at true parameter: {sdw(z, model)}")

latent_model = MaternCovarianceModel()
latent_model.sigma = 1
model = PolynomialModel(latent_model)

sdw = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(sdw)

print(estimator(model, z))
print(f"Likelihood at estimated parameter value: {sdw(z, model)}")
