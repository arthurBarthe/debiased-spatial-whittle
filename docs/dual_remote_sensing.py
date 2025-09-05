# In this notebook we consider an application to dual remote sensing: a univariate field is remotely-sensed, with
# each measurement having a white noise error of a certain level.
# We therefore jointly estimate the noise level for each sensor in parallel with the parameters of the covariance model
# of the sensed field.
# We then proceed to "map" the field using both measurements.

# ##Imports

import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import SquaredExponentialModel, DualRemoteSensing
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import MultivariateSamplerOnRectangularGrid

from debiased_spatial_whittle.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle, Estimator

# ##Grid and model specification

grid = RectangularGrid((64, 64), nvars=2)
latent_model = SquaredExponentialModel(rho=5.0)
model = DualRemoteSensing(latent_model, sigma1=0.1, sigma2=0.2)

# ##Sample a realization

sampler = MultivariateSamplerOnRectangularGrid(model, grid, p=2)
s = sampler()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(s[..., 0], cmap="seismic")
plt.subplot(1, 2, 2)
plt.imshow(s[..., 1], cmap="seismic")
plt.show()

# ##Parameter inference

latent_model = SquaredExponentialModel(rho=1.0)
model = DualRemoteSensing(latent_model, sigma1=0.01, sigma2=0.01)
model.set_param_bounds(dict(sigma1=(0.001, 1.0), sigma2=(0.001, 1.0)))

periodogram = Periodogram()
dbw = MultivariateDebiasedWhittle(periodogram, ExpectedPeriodogram(grid, periodogram))
estimator = Estimator(dbw)
estimator(model, s, opt_callback=lambda *args, **kwargs: print(*args, **kwargs))

# ##Mapping

import numpy as np

xs = np.stack(np.mgrid[:64, :64], -1).reshape(-1, 2)
ys = model.predict(
    (xs, xs), (s[..., 0].reshape((-1, 1)), s[..., 1].reshape((-1, 1))), xs
)
ys = ys.reshape((64, 64))
plt.figure()
plt.imshow(ys, cmap="seismic")
plt.show()
