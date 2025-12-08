from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.univariate import ExponentialModel


def test_covariance_matrix():
    grid = RectangularGrid((8, 8))
    model = ExponentialModel()
    model.rho = 10
    model.sigma = 1
    cov_mat = grid.covariance_matrix(model)
    assert cov_mat.shape == (64, 64)

def test_france_img():
    from debiased_spatial_whittle.grids.old import ImgGrid
    mask_france = ImgGrid((64, 64)).get_new()
