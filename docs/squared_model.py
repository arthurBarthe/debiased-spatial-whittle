import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import SquaredModel, MaternCovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SquaredSamplerOnRectangularGrid

from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator


m, n = 256, 256

grid = RectangularGrid((m, n))

latent_model = MaternCovarianceModel()
latent_model.sigma = 1
latent_model.rho = 24
latent_model.nu = 1.8

model = SquaredModel(latent_model)

sampler = SquaredSamplerOnRectangularGrid(model, grid)
z = sampler()

plt.figure()
plt.imshow(z, cmap='hot')
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
latent_model = MaternCovarianceModel()
model = SquaredModel(latent_model)

sdw = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(sdw)

print(estimator(model, z))