"""
Example script demonstrating the corner_plot_variance_of_estimates function.

This script creates a corner plot showing the distribution of parameter estimates
for a BivariateUniformCorrelation model applied to a SquaredExponentialModel.
"""
from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("torch")
BackendManager.device = "cuda:0"
xp = BackendManager().get_backend()

from debiased_spatial_whittle.models.univariate import ExponentialModel, SquaredExponentialModel, AnisotropicModel, \
    RationalQuadraticModel, Matern32Model
from debiased_spatial_whittle.models.base import SumModel, LogScaleReparameterizedModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation, NuggetModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.diagnostics import corner_plot_variance_of_estimates

# Create grid with nvars=2 for multivariate
grid = RectangularGrid((128, 128), nvars=2)

# Create a SquaredExponentialModel
model = RationalQuadraticModel(rho=8, alpha=1.2, sigma=0.8)
model.set_parameter_bounds("alpha", (0.1, 10.))
model.display_subscript = "S"

model = AnisotropicModel(model, eta=1)
model.set_parameter_bounds("eta", (0.1, 10))
model.fix_parameter("phi")

# Apply BivariateUniformCorrelation
bivariate_model = BivariateUniformCorrelation(model, r=0.5, f=1.0)
bivariate_model.freeze_parameter("f")
bivariate_model.display_subscript = "B"

bivariate_model = NuggetModel(bivariate_model, nugget0=1e-2, nugget1=1e-2)
bivariate_model.freeze_parameter("nugget0")
bivariate_model.freeze_parameter("nugget1")

print(bivariate_model)

# Log Transform on some of the range parameter
bivariate_model_log = LogScaleReparameterizedModel(bivariate_model, (True, True, False, False, True, False, True, True, True))


# Create the corner plot with larger size
fig = corner_plot_variance_of_estimates(bivariate_model_log, grid, width=1200, height=1200, n_sims=100, n_estimates=250)

# Display the figure
fig.show()
