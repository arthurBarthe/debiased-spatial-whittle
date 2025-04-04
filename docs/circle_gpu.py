# In this example, we demonstrate the use of the Spatial Debiased Whittle for inference from data observe within
# a circular region

# ##Backend selection
from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("cupy")

np = BackendManager.get_backend()

# ##Imports

import matplotlib.pyplot as plt
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

# ##Model Specification

model = SquaredExponentialModel(rho=32, sigma=0.9)

# ##Grid specification

m = 512
shape = (m, m)
x_0, y_0, diameter = m // 2, m // 2, m
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
circle = ((x - x_0) ** 2 + (y - y_0) ** 2) <= 1 / 4 * diameter**2
circle = circle * 1.0
grid_circle = RectangularGrid(shape)
grid_circle.mask = circle

# ##Sample generation

sampler = SamplerOnRectangularGrid(model, grid_circle)
z = sampler()

plt.imshow(np.to_cpu(z), origin="lower", cmap="Spectral")
plt.show()

# ##Inference

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_circle, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle, use_gradients=False)

model_est = SquaredExponentialModel(rho=2.0, sigma=1)
model_est.set_param_bounds(dict(rho=(1.0, 100), sigma=(0.1, 10)))

estimate = estimator(model_est, z, opt_callback=lambda *args, **kwargs: print(*args))
print(f"Estimated range parameter: {model_est.rho}")
print(f"Estimated amplitude parameter: {model_est.sigma}")
