# In this notebook we demonstrate the use of the Debiased Spatial Whittle for a simulated random field on an
# incomplete grid, where the sampling region follows the shape of the territory of France.

# ##Imports

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")
import matplotlib.pyplot as plt
import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle


# ##Model specification

model = SquaredExponentialModel()
model.rho = 35
model.sigma = 2
model.nugget = 0.0

# ##Grid specification

shape = (1024 * 1, 1024 * 1)
mask_france = grids.ImgGrid(shape).get_new()
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

model_est = SquaredExponentialModel()
model_est.nugget = None
estimate = estimator(model_est, z)
print(estimate)
