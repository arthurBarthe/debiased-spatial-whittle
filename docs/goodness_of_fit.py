from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend('numpy')


from debiased_spatial_whittle.models.univariate import ExponentialModel, SquaredExponentialModel, NuggetModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import Estimator, DebiasedWhittle
from debiased_spatial_whittle.inference.diagnostics import GoodnessOfFit, generate_goodness_of_fit_plots, \
    generate_goodness_of_fit_plots_3d

model = SquaredExponentialModel(rho=8)
model = NuggetModel(model, nugget=0.01)

m = 256
shape = (m * 1, m * 1)

grid = RectangularGrid(shape)
sampler = SamplerOnRectangularGrid(model, grid)

p_values = []

for i_sample in range(1):
    print(f"---------Sample {i_sample}------------")
    z = sampler()

    periodogram = Periodogram()
    expected_periodogram = ExpectedPeriodogram(grid, periodogram)
    debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
    estimator = Estimator(debiased_whittle)

    def get_model_est():
        model_est = SquaredExponentialModel(rho=1, name="model")
        model_est = NuggetModel(model_est, nugget=0.01, name="nugget")
        model_est.freeze_parameter("nugget_nugget")
        model_est.set_parameter_bounds('model_rho', (1., 100.))
        model_est.set_parameter_bounds('model_sigma', (1 / 10, 10))
        return model_est

    model_est = get_model_est()
    estimate = estimator(model_est, z)
    print(estimate)

    # we carry out some goodness-of-fit analysis
    gof = GoodnessOfFit(model_est, grid, z, n_bins=500)
    gof.get_model_est = get_model_est

    fig = generate_goodness_of_fit_plots(gof)
    fig.show()

    chi, p_value = gof.compute_diagnostic_statistic()
    print(chi, p_value)
    p_value = gof.p_value(chi, n_sim=100)
    print(p_value)
    p_values.append(p_value)