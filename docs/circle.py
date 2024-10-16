# In this example, we demonstrate the use of the Spatial Debiased Whittle for inference from data observe within
# a circular region

# ##Imports

import numpy as np
import matplotlib.pyplot as plt
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

# ##Model Specification

model = SquaredExponentialModel()
model.rho = 20
model.sigma = 2
model.nugget = 0.025

# ##Grid specification

m = 256
shape = (m * 1, m * 1)
x_0, y_0, diameter = m // 2, m // 2, m
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
circle = ((x - x_0)**2 + (y - y_0)**2) <= 1 / 4 * diameter**2
circle = circle * 1.
grid_circle = RectangularGrid(shape)
grid_circle.mask = circle

# ##Sample generation

sampler = SamplerOnRectangularGrid(model, grid_circle)
z = sampler()

# ##Inference

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_circle, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=True)

model_est = SquaredExponentialModel()
model_est.nugget = None
estimate = estimator(model_est, z)
print(estimate)

plt.imshow(z, origin='lower', cmap='Spectral')
plt.show()