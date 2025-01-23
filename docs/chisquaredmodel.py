import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import (
    ChiSquaredModel,
    MaternCovarianceModel,
    SquaredExponentialModel,
)
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import ChiSquaredSamplerOnRectangularGrid

from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator


m, n = 256, 256

grid = RectangularGrid((m, n))

latent_model = SquaredExponentialModel()
latent_model.sigma = 1
latent_model.rho = 16
latent_model.nugget = 0.01

model = ChiSquaredModel(latent_model)
model.dof_1 = 3

sampler = ChiSquaredSamplerOnRectangularGrid(model, grid)
z = sampler()

plt.figure()
plt.imshow(z, cmap="hot")
plt.show()

# compute periodogram and show residuals
periodogram = Periodogram()
per = periodogram(z)

expected_periodogram = ExpectedPeriodogram(grid, periodogram)
ep = expected_periodogram(model)

import numpy as np
from numpy.fft import fftshift

plt.figure()
plt.imshow(10 * np.log10(fftshift((per / ep))))
plt.colorbar()
plt.show()

print(np.mean(per / ep))


# estimation
latent_model = SquaredExponentialModel()
model = ChiSquaredModel(latent_model)
model.dof_1 = None

sdw = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(sdw)

print(estimator(model, z))
