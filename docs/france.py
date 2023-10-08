from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('torch')

import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

model = SquaredExponentialModel()
model.rho = 35
model.sigma = 1
model.nugget = 0.025

shape = (1024 * 1, 1024 * 1)
mask_france = grids.ImgGrid(shape).get_new()
grid_france = RectangularGrid(shape)
grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)

z = sampler()

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=True)

model_est = SquaredExponentialModel()
model_est.nugget = model.nugget.value
estimate = estimator(model_est, z, opt_callback=lambda *args, **kargs: print(args, kargs))
print(estimate)

plt.imshow(z, origin='lower', cmap='Spectral')
plt.show()