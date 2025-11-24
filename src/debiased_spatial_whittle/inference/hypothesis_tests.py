from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()
inv = BackendManager.get_inv()
from numpy.linalg import eig

from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.inference.likelihood import (
    DebiasedWhittle,
    MultivariateDebiasedWhittle,
    Estimator,
)
from typing import Union
from collections import namedtuple


class TemporaryModel:
    def __init__(self, model: CovarianceModel, fixed_parameters: dict):
        self.model = model
        self.free_params = model.free_params
        self.fixed_parameters = fixed_parameters

    def __enter__(self):
        for p_name, p_value in self.fixed_parameters.items():
            self.model.params[p_name].value = p_value
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p_name in self.free_params.names:
            self.model.params[p_name].value = None


HypothesisTestResult = namedtuple(
    "HypothesisTestResult", ("lkh_ratio", "max_eig", "p_value", "test_result")
)


class FixedParametersHT:
    """
    Class for the design of a Hypothesis Test that involves nested models where the null
    hypothesis is obtained by fixing some parameters of a full model.
    """

    def __init__(
        self,
        full_model: CovarianceModel,
        null_parameters: dict,
        likelihood: Union[DebiasedWhittle, MultivariateDebiasedWhittle],
    ):
        self.full_model = full_model
        self.null_parameters = null_parameters
        self.likelihood = likelihood
        self.estimator = Estimator(likelihood)

    def _sample_generalized_chi_squared(self, lambdas):
        n = lambdas.shape[0]
        zs = xp.random.randn(1000, n)
        lambdas = lambdas.reshape((n, 1))
        sample = xp.matmul(zs ** 2, lambdas)
        return sample

    def __call__(self, z: xp.array, level=0.05):
        params = self.full_model.free_params
        with TemporaryModel(self.full_model, dict()) as full_model:
            self.estimator(full_model, z)
            lkh_full = self.likelihood(z, full_model)
        with TemporaryModel(self.full_model, self.null_parameters) as null_model:
            params_null = null_model.free_params
            self.estimator(null_model, z)
            lkh_null = self.likelihood(z, null_model)
            h_null = self.likelihood.fisher(null_model, params_null)
            h_full = self.likelihood.fisher(null_model, params)
            j_full = self.likelihood.jmatrix_sample(null_model, params)
        lkh_ratio = -(lkh_full - lkh_null) * z.shape[0] * z.shape[1]
        h_full_inv = inv(h_full)
        h_null_inv = xp.pad(inv(h_null), ((1, 0), (1, 0)))
        w_matrix = (
                z.shape[0] * z.shape[1] / 2 * xp.matmul(j_full, -h_null_inv + h_full_inv)
        )
        eigenvalues = eig(w_matrix)[0]
        sample_dist = self._sample_generalized_chi_squared(eigenvalues)
        test = lkh_ratio < xp.quantile(sample_dist, 1 - level)
        p_value = xp.mean(lkh_ratio <= sample_dist)
        print(test)
        return HypothesisTestResult(
            lkh_ratio, xp.max(xp.abs(eigenvalues)), p_value, test
        )
