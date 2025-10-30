# In this notebook we demonstrate how we can estimate the parameters of a covariance model
# and use the fitted model for interpolation on made-up data.
import numpy as np

# ##Imports

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")
import matplotlib.pyplot as plt
import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models.univariate import SquaredExponentialModel, NuggetModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.grids.old import ImgGrid


# ##Model specification

model = SquaredExponentialModel(rho=7, sigma=0.9)

# ##Grid specification

shape = (128, 128)
mask_france = ImgGrid(shape).get_new().astype(bool)
grid_france = RectangularGrid(shape)
grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)

# ##Sample generation

z = sampler()
plt.figure()
plt.imshow(z, origin="lower", cmap="RdBu")
plt.title("full sample")
plt.show()


# ## Observed sample
def add_missing_circle(mask, centre, radius):
    m, n = mask.shape
    xs, ys = np.mgrid[:m, :n]
    sel = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 <= radius**2
    mask[sel] = 0
    return xs[sel], ys[sel], sel


xs_pred, ys_pred, sel_pred = add_missing_circle(mask_france, (64, 64), 15)

z_obs = z * mask_france
plt.figure()
plt.imshow(z_obs, origin="lower", cmap="RdBu")
plt.title("partial sample")
plt.show()

# ##Inference

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle)

model_est = SquaredExponentialModel()
model_est.sigma = 0.9
model_est.fix_parameter("sigma")
estimate = estimator(model_est, z_obs)
print(estimate.rho)


# ## Interpolation

# we add a small nugget to the model for computational stability
model_est = NuggetModel(model_est, nugget=0.001)

xs, ys = np.mgrid[: shape[0], : shape[1]]
xs_obs, ys_obs = xs[mask_france], ys[mask_france]
x_obs = np.stack((xs_obs, ys_obs), -1)
x_pred = np.stack((xs_pred, ys_pred), -1)
y_pred = model_est.predict(x_obs, z_obs[mask_france].reshape(-1, 1), x_pred)

z_obs[sel_pred] = y_pred.flatten()
plt.figure()
plt.imshow(z_obs, origin="lower", cmap="RdBu")
plt.title("mapped sample")
plt.show()
