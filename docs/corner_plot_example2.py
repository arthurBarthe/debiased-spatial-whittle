"""
Example script demonstrating the corner_plot_variance_of_estimates function.

This script creates a corner plot showing the distribution of parameter estimates
for a BivariateUniformCorrelation model applied to a SquaredExponentialModel.
"""
from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("torch")
BackendManager.device = "cuda:0"
xp = BackendManager().get_backend()

from debiased_spatial_whittle.models.univariate import SquaredExponentialModel, AnisotropicModel, NuggetModel
from debiased_spatial_whittle.models.base import SumModel, LogScaleReparameterizedModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.diagnostics import corner_plot_variance_of_estimates

# Create a SquaredExponentialModel
model = SquaredExponentialModel(rho=16, sigma=1)
model.set_parameter_bounds("rho", (1., 100.))
model.set_parameter_bounds("sigma", (0.1, 10))

model = AnisotropicModel(model, eta=0.5)
model.set_parameter_bounds("eta", (0.1, 10))
model.fix_parameter("phi")

model = NuggetModel(model, nugget=0.005)
model.set_parameter_bounds("nugget", (1e-5, 1e-2))

print(model)

bivariate_model_log = LogScaleReparameterizedModel(model, (True, True, False, True, True))

# Create grid with nvars=2 for multivariate
grid = RectangularGrid((128, 128))

# Create the corner plot with larger size
fig = corner_plot_variance_of_estimates(bivariate_model_log, grid, width=1200, height=1200, n_sims=100, n_estimates=100)

# Display the figure
fig.show()
