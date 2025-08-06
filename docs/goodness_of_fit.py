# imports

import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    Matern32Model,
    Matern52Model,
)
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.diagnostics import GoodnessOfFit
from debiased_spatial_whittle.utils import plot_fourier_values

# ## True model and sampler

model = Matern32Model(rho=8, sigma=0.9)

m = 256
shape = (m, m)

grid = RectangularGrid(shape)
sampler = SamplerOnRectangularGrid(model, grid)

z = sampler()

# ## Model inference from sample

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle)


def get_model_est():
    model_est = Matern32Model()
    model_est.set_param_bounds(dict(rho=(1.0, m), sigma=(0.1, 10)))
    return model_est


model_est = get_model_est()
estimate = estimator(model_est, z)
print("Rho estimate: ", estimate.rho)

# ## Goodness-of-fit analysis using uniform residuals

gof = GoodnessOfFit(model_est, grid, z, n_bins=500)
gof.plot()

gof.bootstrap = False
gof.get_model_est = get_model_est
chi, p_value = gof.compute_diagnostic_statistic()
p_value = gof.p_value(chi)
print("p-value: ", p_value)

# ## Goodness-of-fit using ratio residuals

from debiased_spatial_whittle.diagnostics import GoodnessOfFitSimonsOlhede

gof = GoodnessOfFitSimonsOlhede(model_est, grid, z)
print(gof.compute_diagnostic_statistic())
gof.plot()
