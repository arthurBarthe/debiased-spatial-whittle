import numpy as np
import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

shape = (620 * 1, 620 * 1)

model = SquaredExponentialModel()
model.rho = 16
model.sigma = 1
model.nugget = 0.0

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
estimator = Estimator(debiased_whittle, use_gradients=True)

model_est = SquaredExponentialModel()
model_est.nugget = None
estimate = estimator(model_est, z)
print(estimate)

z[mask_france == 0] = np.nan
plt.imshow(z, origin="lower", cmap="Spectral")
plt.show()
