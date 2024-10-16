from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel

def test_covariance_matrix():
    grid = RectangularGrid((8, 8))
    model = ExponentialModel()
    model.rho = 10
    model.sigma = 1
    cov_mat = grid.covariance_matrix(model)
    assert cov_mat.shape == (64, 64)
