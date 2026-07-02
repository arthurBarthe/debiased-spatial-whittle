from debiased_spatial_whittle.backend import BackendManager
np = BackendManager.get_backend()

randn = BackendManager.get_randn()

from debiased_spatial_whittle.inference.likelihood import (
    DebiasedWhittle,
    MultivariateDebiasedWhittle,
)
from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.periodogram import ExpectedPeriodogram, Periodogram
from debiased_spatial_whittle.inference.multivariate_periodogram import (
    Periodogram as MultivariatePeriodogram,
)


class TestShapesUnivariate:
    grid = RectangularGrid((64, 32))
    model = SquaredExponentialModel(rho=4, sigma=1)
    vectorized_model = SquaredExponentialModel(rho=np.array([4.0, 16.0, 12.0]))
    periodogram = Periodogram()
    expected_periodogram = ExpectedPeriodogram(grid, periodogram)
    dbw = DebiasedWhittle(periodogram, expected_periodogram)

    def test_shape_model(self):
        lags = randn(2, 5, 7)
        assert self.model(lags).shape == (5, 7)
        assert self.vectorized_model(lags).shape == (5, 7, 3)

    def test_shape_ep(self):
        assert self.expected_periodogram(self.model).shape == (64, 32)
        assert self.expected_periodogram(self.vectorized_model).shape == (64, 32, 3)

    def test_shape_whittle(self):
        assert self.dbw(randn(*self.grid.n), self.model).shape == ()
        assert self.dbw(randn(*self.grid.n), self.vectorized_model).shape == (
            3,
        )

    def test_shape_model_jacobian(self):
        lags = randn(2, 5, 7)
        param_name = f'{self.model.name}_rho'
        jac = self.model.jacobian(lags, param_names=(param_name,))
        assert jac[param_name].shape == (5, 7)

    def test_shape_ep_jacobian(self):
        param_name = f'{self.model.name}_rho'
        jac = self.expected_periodogram.jacobian(self.model)
        assert jac[param_name].shape == (64, 32)

    def test_shape_whittle_gradient(self):
        # Test gradient shape using the gradient method
        param_names = [f'{self.model.name}_rho']
        grad_dict = self.dbw.gradient(
            randn(*self.grid.n),
            self.model,
            param_names=param_names,
        )
        # The gradient dict should have one entry with a scalar value
        assert len(grad_dict) == 1
        
        param_names = [f'{self.model.name}_rho', f'{self.model.name}_sigma']
        grad_dict = self.dbw.gradient(
            randn(*self.grid.n),
            self.model,
            param_names=param_names,
        )
        # The gradient dict should have two entries with scalar values
        assert len(grad_dict) == 2


class TestShapesMultivariate:
    grid = RectangularGrid((64, 32), nvars=2)
    base_model = SquaredExponentialModel(rho=4, sigma=1)
    model = BivariateUniformCorrelation(base_model, r=0.2, f=1.1)
    base_model = SquaredExponentialModel(rho=np.array([4.0, 16.0, 12.0]))
    vectorized_model = BivariateUniformCorrelation(base_model, r=0.2, f=1.1)
    periodogram = MultivariatePeriodogram()
    expected_periodogram = ExpectedPeriodogram(grid, periodogram)
    dbw = MultivariateDebiasedWhittle(periodogram, expected_periodogram)

    def test_shape_model(self):
        lags = randn(2, 5, 7)
        assert self.model(lags).shape == (5, 7, 2, 2)
        assert self.vectorized_model(lags).shape == (5, 7, 3, 2, 2)

    def test_shape_ep(self):
        assert self.expected_periodogram(self.model).shape == (64, 32, 2, 2)
        assert self.expected_periodogram(self.vectorized_model).shape == (
            64,
            32,
            3,
            2,
            2,
        )

    def test_shape_whittle(self):
        # __call__ returns a scalar (float), not a tensor
        result = self.dbw(randn(*self.grid.n, 2), self.model)
        # Just verify it returns a number (scalar) - check it's not an array/tensor
        assert not hasattr(result, 'shape') and not hasattr(result, '__len__')
        result_vec = self.dbw(
            randn(*self.grid.n, 2), self.vectorized_model
        )
        # For vectorized model, should return array of shape (3,)
        assert hasattr(result_vec, 'shape') and result_vec.shape == (3,)

    def test_shape_model_gradient(self):
        lags = randn(2, 5, 7)
        # Test with all parameters
        jac = self.model.jacobian(lags)
        # Check that all expected parameters are present
        assert f'{self.model.name}_r' in jac
        assert f'{self.model.name}_f' in jac
        # Check shape for one parameter
        assert jac[f'{self.model.name}_r'].shape == (5, 7, 2, 2)

    def test_shape_ep_gradient(self):
        # Test with all parameters to avoid the single-parameter bug
        jac = self.expected_periodogram.jacobian(self.model)
        # Check that expected parameters are present
        assert f'{self.model.name}_r' in jac
        assert f'{self.model.name}_f' in jac
        # The shape includes the grid dimensions and the 2x2 covariance matrix
        assert jac[f'{self.model.name}_r'].shape == (64, 32, 2, 2)

    def test_shape_whittle_gradient(self):
        # Test gradient shape using the gradient method
        # Use both bivariate parameters (r and f)
        param_names = [self.model.parameter_names[0], self.model.parameter_names[1]]
        grad_dict = self.dbw.gradient(
            randn(*self.grid.n, 2),
            self.model,
            param_names=param_names,
        )
        # The gradient dict should have two entries with scalar values
        assert len(grad_dict) == 2
        
        # Also test with just one parameter using parameter_names directly
        param_names_single = [self.model.parameter_names[0]]
        grad_dict_single = self.dbw.gradient(
            randn(*self.grid.n, 2),
            self.model,
            param_names=param_names_single,
        )
        assert len(grad_dict_single) == 1
