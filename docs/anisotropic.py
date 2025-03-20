from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")

np = BackendManager.get_backend()

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import (
    SquaredExponentialModel,
    Matern32Model,
    Matern52Model,
    NuggetModel,
    AnisotropicModel,
)
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.least_squares import LeastSquareEstimator
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator

import matplotlib.pyplot as plt

# ## Set up grid and model
grid = RectangularGrid((256, 256))
model = SquaredExponentialModel(rho=12.0)
model = AnisotropicModel(model, eta=1.7, phi=np.pi / 4)
model = NuggetModel(model, nugget=1e-2)
model

# ## Sample from the model
sampler = SamplerOnRectangularGrid(model, grid)
data = sampler()

plt.figure()
plt.pcolor(data, cmap="RdBu")
plt.show()

# ## Set up estimation
periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)

model_est = SquaredExponentialModel(rho=10.0)
model_est.param.rho.bounds = (5, 100)
model_est.param.sigma.bounds = (0.1, 10)
model_est = AnisotropicModel(model_est)

model_est_ = NuggetModel(model_est, nugget=1e-3)

debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

# ## Carry out least square fit
least_square = LeastSquareEstimator(periodogram, expected_periodogram)
least_square(data, model_est_)
print(model_est_.free_parameter_values_to_array_deep())
model_est_

# ## Carry out Debiased Whittle fit
estimator = Estimator(debiased_whittle)
estimator(model_est_, data)
print(model_est_.free_parameter_values_to_array_deep())
model_est_
