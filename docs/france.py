from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('numpy')

import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternCovarianceModel, SpectralMatern
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

model = SpectralMatern()
model.rho = 25
model.sigma = 1
model.nu = 2.3
#model.nugget = 0.025

shape = (512 * 1, 512 * 1)
mask_france = grids.ImgGrid(shape).get_new()
grid_france = RectangularGrid(shape)
grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)

z = sampler()

plt.imshow(z, origin='lower', cmap='Spectral', vmin=-2, vmax=2)
plt.show()

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=False, optim_options=dict(maxfun=100, maxiter=5, no_local_search=False), method='dual_annealing')

model_est = MaternCovarianceModel()
#model_est.nugget = None
estimate = estimator(model_est, z, opt_callback=lambda *args, **kargs: print(args))

print(estimate)

