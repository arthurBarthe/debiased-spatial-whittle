import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternCovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

model = MaternCovarianceModel()
model.rho = 35
model.sigma = 2
model.nu = 1.5
#model.nugget = 0.025

shape = (1024 * 1, 1024 * 1)
mask_france = grids.ImgGrid(shape).get_new()
grid_france = RectangularGrid(shape)
grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)

z = sampler()

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=False)

model_est = MaternCovarianceModel()
model_est.nu = 1.5
#model_est.nugget = None
estimate = estimator(model_est, z, opt_callback=lambda *args, **kargs: print(args))
print(estimate)

plt.imshow(z, origin='lower', cmap='Spectral')
plt.show()