# Here we demonstrate how we can predict the variance of our estimates. This requires to compute gradients of
# the covariance function with respect to its parameters, which relies on pytorch automatic gradient calculation.

# ##Imports

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("numpy")
xp = BackendManager.get_backend()

import matplotlib.pyplot as plt
from debiased_spatial_whittle.models.univariate import SquaredExponentialModel, NuggetModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import Estimator, DebiasedWhittle

# ##Model Specification
model = NuggetModel(SquaredExponentialModel(rho=10., sigma=0.9),
                    nugget=0.1)

# ##Grid specification

m = 128
shape = (m * 1, m * 1)
x_0, y_0, diameter = m // 2, m // 2, m
x, y = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing="ij")
circle = ((x - x_0) ** 2 + (y - y_0) ** 2) <= 1 / 4 * diameter**2
circle = circle * 1.0
grid_circle = RectangularGrid(shape)
grid_circle.mask = circle

# ##Sample generation

sampler = SamplerOnRectangularGrid(model, grid_circle)
z = sampler()

# ##Inference

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_circle, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=False)

model_est = NuggetModel(SquaredExponentialModel(rho=1., sigma=1.),
                    nugget=0.01)
estimate = estimator(model_est, z)
print("Estimated nugget: ", model_est.nugget)
print("Estimated range parameter: ", model_est.children[0].rho)
print("Estimated amplitude parameter: ", model_est.children[0].sigma)

plt.imshow(z, origin="lower", cmap="Spectral")
plt.show()

# ##obtain the predicted covariance matrix of parameter estimates
jmat = debiased_whittle.jmatrix_sample(model, model.param_objects())
v = debiased_whittle.variance_of_estimates(model, model.param_objects(), jmat)
print(v)