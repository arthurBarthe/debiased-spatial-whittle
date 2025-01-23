import matplotlib.pyplot as plt

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import (
    TMultivariateModel,
    MaternCovarianceModel,
    ExponentialModel,
)
from debiased_spatial_whittle.simulation import (
    TSamplerOnRectangularGrid,
    SamplerOnRectangularGrid,
)
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram

latent_model = ExponentialModel()
# latent_model.nu = 10.5
latent_model.rho = 30
latent_model.sigma = 1
latent_model.nugget = 0.01

model = TMultivariateModel(latent_model)
model.nu_1 = 5
print(model.params)

###
# model = latent_model

grid = RectangularGrid((512, 512))
sampler = TSamplerOnRectangularGrid(model, grid)

# sampler = SamplerOnRectangularGrid(model, grid)


z = sampler()
plt.figure()
plt.imshow(z, cmap="bwr", vmin=-3, vmax=3)

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
