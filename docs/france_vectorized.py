# In this notebook we demonstrate the use of the Debiased Spatial Whittle for a simulated random field on an
# incomplete grid, where the sampling region follows the shape of the territory of France.

# ##Imports

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")
np = BackendManager.get_backend()

import matplotlib.pyplot as plt
import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.grids.old import ImgGrid


# ##Model specification

model = SquaredExponentialModel(rho=15, sigma=0.9)

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

model_est = SquaredExponentialModel()
model_est.sigma = 0.9
model_est.fix_parameter("sigma")

model.rho = np.arange(5, 20)
print(debiased_whittle(z, model))
