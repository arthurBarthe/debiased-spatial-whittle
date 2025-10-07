import numpy as np
from numpy.testing import assert_allclose

from debiased_spatial_whittle.likelihood import (
    DebiasedWhittle,
    MultivariateDebiasedWhittle,
)
from debiased_spatial_whittle.simulation import SamplerBUCOnRectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.models import BivariateUniformCorrelation
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram, Periodogram
from debiased_spatial_whittle.multivariate_periodogram import (
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
        lags = np.random.randn(2, 5, 7)
        assert self.model(lags).shape == (5, 7)
        assert self.vectorized_model(lags).shape == (5, 7, 3)

    def test_shape_ep(self):
        assert self.expected_periodogram(self.model).shape == (64, 32)
        assert self.expected_periodogram(self.vectorized_model).shape == (64, 32, 3)

    def test_shape_whittle(self):
        assert self.dbw(np.random.randn(*self.grid.n), self.model).shape == ()
        assert self.dbw(np.random.randn(*self.grid.n), self.vectorized_model).shape == (
            3,
        )

    def test_shape_model_gradient(self):
        lags = np.random.randn(2, 5, 7)
        params_gradient = [
            self.model.param.rho,
        ]
        assert self.model.gradient(lags, params_gradient).shape == (5, 7, 1)

    def test_shape_ep_gradient(self):
        params_gradient = [
            self.model.param.rho,
        ]
        assert self.expected_periodogram.gradient(
            self.model, params_gradient
        ).shape == (64, 32, 1)

    def test_shape_whittle_gradient(self):
        params_gradient = [
            self.model.param.rho,
        ]
        assert self.dbw(
            np.random.randn(*self.grid.n),
            self.model,
            params_for_gradient=params_gradient,
        )[1].shape == (1,)
        params_gradient = [self.model.param.rho, self.model.param.sigma]
        assert self.dbw(
            np.random.randn(*self.grid.n),
            self.model,
            params_for_gradient=params_gradient,
        )[1].shape == (2,)


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
        lags = np.random.randn(2, 5, 7)
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
        assert self.dbw(np.random.randn(*self.grid.n, 2), self.model).shape == ()
        assert self.dbw(
            np.random.randn(*self.grid.n, 2), self.vectorized_model
        ).shape == (3,)

    def test_shape_model_gradient(self):
        lags = np.random.randn(2, 5, 7)
        params_gradient = [
            self.model.param.r,
        ]
        assert self.model.gradient(lags, params_gradient).shape == (5, 7, 1, 2, 2)

    def test_shape_ep_gradient(self):
        params_gradient = [
            self.model.param.r,
        ]
        assert self.expected_periodogram.gradient(
            self.model, params_gradient
        ).shape == (64, 32, 1, 2, 2)

    def test_shape_whittle_gradient(self):
        params_gradient = [
            self.model.param.r,
        ]
        assert self.dbw(
            np.random.randn(*self.grid.n, 2),
            self.model,
            params_for_gradient=params_gradient,
        )[1].shape == (1,)
        params_gradient = [self.model.param.r, self.model.param.f]
        assert self.dbw(
            np.random.randn(*self.grid.n, 2),
            self.model,
            params_for_gradient=params_gradient,
        )[1].shape == (2,)
