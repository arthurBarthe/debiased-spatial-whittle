# In this example, we demonstrate the use of the Spatial Debiased Whittle for inference from data observe within
# a circular region

# ##Imports

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("numpy")
xp = BackendManager.get_backend()

import matplotlib.pyplot as plt
from debiased_spatial_whittle.models.univariate import SquaredExponentialModel, ExponentialModel, NuggetModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.models import TaperedCovarianceModel, WendlandTaper


# ##Model Specification

model = SquaredExponentialModel(rho=8, sigma=0.9, name="basemodel")

# ##Grid specification

m = 256
shape = (m * 1, m * 1)
grid = RectangularGrid(shape)

# ##Sample generation

sampler = SamplerOnRectangularGrid(model, grid)
sample = sampler()

# ##Inference

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=False)

model_est = model.copy()
model_est.rho = 1.
model_est.set_parameter_bounds("rho", (1., 100.))
model_est.fix_parameter("basemodel_sigma")

taper = WendlandTaper(range=64, type="C6")
model_est = TaperedCovarianceModel(model_est, taper=taper)


estimate = estimator(model_est, sample)
print(model)
print(model_est)

plt.imshow(sample, origin="lower", cmap="Spectral")
plt.show()

# plot of taper
x = xp.arange(0, 256, 0.1)
y = taper(x.reshape((1, -1)))
plt.figure()
plt.plot(x, y)
plt.show()

# figure with true expected periodogram and expected periodogram of the tapered covariance
true_ep = expected_periodogram(model)
tapered_ep = expected_periodogram(TaperedCovarianceModel(model, taper=taper))

fftshift, ifftshift = BackendManager.get_fftshift_methods()
plt.figure()
plt.subplot(131)
plt.imshow(10 * xp.log10(fftshift(true_ep)))
plt.subplot(132)
plt.imshow(10 * xp.log10(fftshift(tapered_ep)))
plt.subplot(133)
plt.imshow(fftshift(tapered_ep / true_ep))
plt.show()
