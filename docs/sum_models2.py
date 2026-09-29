import logging
logging.basicConfig(level=logging.ERROR)
from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("torch")
BackendManager.device = "cuda:0"

xp = BackendManager.get_backend()

from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.base import Sum2Models, SigmoidReparameterizedModel, LogScaleReparameterizedModel
from debiased_spatial_whittle.models.univariate import (
    ExponentialModel,
    SquaredExponentialModel,
    NuggetModel,
)

from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.least_squares import LeastSquareEstimator
from debiased_spatial_whittle.inference.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.inference.diagnostics import corner_plot_variance_of_estimates



import matplotlib.pyplot as plt

# ##Set up grid and model

grid = RectangularGrid((512, 512))
model_1 = ExponentialModel(rho=5.0)
model_2 = SquaredExponentialModel(rho=16.0)
true_model = Sum2Models(model_1, model_2, theta=1.1)
print(true_model)

# ##Sample from the model

sampler = SamplerOnRectangularGrid(true_model, grid)
data = sampler()

plt.figure()
plt.pcolor(data.values, cmap="RdBu")
plt.show()

# ##Inference

model_1 = ExponentialModel(rho=8.0, name="exp")
model_1.set_parameter_bounds("rho", (1, 100))
model_1.freeze_parameter("sigma")

model_2 = SquaredExponentialModel(rho=8.0, name="sqexp")
model_2.set_parameter_bounds("rho", (1, 100))
model_2.freeze_parameter("sigma")


model = Sum2Models(model_1, model_2)
model = LogScaleReparameterizedModel(model, sel=(False, True, False, True, False))
print(model.base_model)

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

least_squares = LeastSquareEstimator(periodogram, expected_periodogram, verbose=2)
least_squares(data, model)

estimator = Estimator(debiased_whittle)

estimator(model, data, opt_callback=lambda *args, **kwargs: print(*args, **kwargs))
print(model.base_model)

fig = corner_plot_variance_of_estimates(model, grid, width=1200, height=1200, n_sims=100)
fig.show()