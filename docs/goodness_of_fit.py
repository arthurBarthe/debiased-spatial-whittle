import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternCovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.diagnostics import GoodnessOfFit

model = ExponentialModel()
model.nugget = 0.1
model.rho = 10
model.sigma = 1.

m = 256
shape = (m * 1, m * 1)

grid_circle = RectangularGrid(shape)
sampler = SamplerOnRectangularGrid(model, grid_circle)

p_values = []

fig = plt.figure()
ax = fig.add_subplot()

for i_sample in range(1):
    print(f'---------Sample {i_sample}------------')
    z = sampler()

    periodogram = Periodogram()
    expected_periodogram = ExpectedPeriodogram(grid_circle, periodogram)
    debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
    estimator = Estimator(debiased_whittle, use_gradients=False)

    def get_model_est():
        model_est = SquaredExponentialModel()
        model_est.nugget = None
        return model_est

    model_est = get_model_est()
    estimate = estimator(model_est, z)
    print(estimate)

    # we carry out some goodness-of-fit analysis
    gof = GoodnessOfFit(model_est, grid_circle, z, n_bins=500)

    residuals = gof.compute_residuals(z, model)
    plt.figure()
    plt.hist(residuals.flatten())

    gof.bootstrap = False
    gof.get_model_est = get_model_est
    chi, p_value = gof.compute_diagnostic_statistic()
    print(chi, p_value)
    p_value = gof.p_value(chi)
    print(p_value)
    p_values.append(p_value)

    # update plot
    ax.clear()
    ax.hist(p_values, bins=np.linspace(0, 1, 20))
    plt.pause(0.1)
    plt.show(block=False)
plt.show(block=True)
