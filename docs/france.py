# In this notebook we demonstrate the use of the Debiased Spatial Whittle for a simulated random field on an
# incomplete grid, where the sampling region follows the shape of the territory of France.

# ##Imports


import matplotlib.pyplot as plt
import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models.spectral import SpectralMatern
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.grids.old import ImgGrid

# ##Model specification

model = SpectralMatern(rho=15, nu=2.0)

# ##Grid specification

shape = (512, 512)
mask_france = ImgGrid(shape).get_new()
grid_france = RectangularGrid(shape)
grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)

# ##Sample generation

z = sampler()
plt.figure()
plt.imshow(z, origin="lower", cmap="RdBu")
plt.show()

# ##Inference

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle)

model_est = SpectralMatern()
model_est.set_param_bounds(dict(rho=(3, 50), nu=(0.5, 5)))
estimate = estimator(model_est, z, opt_callback=lambda *args, **kwargs: print(*args))
print(estimate.rho)
