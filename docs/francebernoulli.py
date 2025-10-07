import numpy as np
import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.sdf_models import SpectralMatern
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

shape = (620, 620)

model = SquaredExponentialModel(rho=16, sigma=1)

p_obs = 0.9
mask_bernoulli = np.random.rand(*shape) <= p_obs

mask_france = grids.ImgGrid(shape).get_new() * mask_bernoulli
print(f"Number of observations: {np.sum(mask_france)}")
grid_france = RectangularGrid(shape)
grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)

z = sampler()

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle)

model_est = SquaredExponentialModel()
estimate = estimator(model_est, z)
print(estimate.rho)

z[mask_france == 0] = np.nan
plt.imshow(z, origin="lower", cmap="Spectral")
plt.show()
