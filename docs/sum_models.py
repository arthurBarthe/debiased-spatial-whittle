from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("cupy")

np = BackendManager.get_backend()

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import (
    ExponentialModel,
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

# ##Set up grid and model

grid = RectangularGrid((512, 512))
model_1 = ExponentialModel(rho=5.0, sigma=np.sqrt(1 / 3))
model_2 = SquaredExponentialModel(rho=32.0, sigma=np.sqrt(2 / 3))
model = model_1 + model_2
model = NuggetModel(model, nugget=1e-2)
model

# ##Sample from the model

sampler = SamplerOnRectangularGrid(model, grid)
data = sampler()

plt.figure()
plt.pcolor(data.get(), cmap="RdBu")
plt.show()

# ##Inference

model_1 = ExponentialModel(rho=1.0, sigma=1 / 1.41)
model_2 = SquaredExponentialModel(rho=2.0, sigma=1 / 1.41)
model = model_1 + model_2
model = NuggetModel(model, nugget=1e-2)
model.fix_parameter("nugget")

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

least_squares = LeastSquareEstimator(periodogram, expected_periodogram, verbose=2)
least_squares(data, model)

estimator = Estimator(debiased_whittle)

estimator(model, data, opt_callback=lambda *args, **kwargs: print(*args, **kwargs))
