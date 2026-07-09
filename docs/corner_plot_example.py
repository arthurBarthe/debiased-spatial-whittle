"""
Example script demonstrating the corner_plot_variance_of_estimates function.

This script creates a corner plot showing the distribution of parameter estimates
for a BivariateUniformCorrelation model applied to a SquaredExponentialModel.
"""
from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("torch")
BackendManager.device = "cpu"
xp = BackendManager().get_backend()

from debiased_spatial_whittle.models.univariate import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.models.base import SumModel, LogScaleReparameterizedModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.diagnostics import corner_plot_variance_of_estimates

# Create a SquaredExponentialModel
model = ExponentialModel(rho=3, sigma=0.8)

# Apply BivariateUniformCorrelation
bivariate_model = BivariateUniformCorrelation(model, r=0.5, f=1.0)

# Log Transform on some of the range parameter
bivariate_model_log = LogScaleReparameterizedModel(bivariate_model, (False, False, True, True))

# Create grid with nvars=2 for multivariate
grid = RectangularGrid((64, 64), nvars=2)

# Create the corner plot with larger size
fig = corner_plot_variance_of_estimates(bivariate_model_log, grid, width=1200, height=1200, n_sims=10)

# Display the figure
fig.show()
