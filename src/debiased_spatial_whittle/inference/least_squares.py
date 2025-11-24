"""
This module provides a method for a least-squares fit of the expected periodogram
to the periodogram. This can be used for instance to obtain an initial guess
before using Debiased Whittle estimation.
"""

from typing import Callable

from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()

from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.models.base import CovarianceModel
from scipy.optimize import least_squares


class LeastSquareEstimator:
    """
    Implements Least-Square estimation using scipy.optimize's non-linear least-square optimization algorithm.
    Can be used for instanced to obtain initial guesses for the optimization of the Debiased Spatial Whittle.

    Attributes
    ----------
    periodogram
        Periodogram applied to the data

    expected_periodogram
        Expected periodogram used for the fit
    """

    def __init__(
        self,
        periodogram: Periodogram,
        expected_periodogram: ExpectedPeriodogram,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        periodogram
            Periodogram object that will be applied to the data
        expected_periodogram
            Expected periodogram object
        verbose
            Verbosity level passed on to the least_squares function of scipy.optimize
        """
        self.periodogram = periodogram
        self.expected_periodogram = expected_periodogram
        self.verbose = verbose

    def __call__(self, data: xp.array, model: CovarianceModel) -> CovarianceModel:
        """
        Carries out the Least-Square estimation.

        Parameters
        ----------
        data
            Sampled random field
        model
            Covariance model
        Returns
        -------
        model
            Fitted covariance model
        """
        x0 = model.free_parameter_values_to_array_deep()
        bounds = xp.array(list(zip(*model.free_parameter_bounds_to_list_deep())))

        # convert to cpu
        x0 = xp.to_cpu(x0)
        bounds = xp.to_cpu(bounds)

        least_squares(
            self._get_opt_func(data, model),
            x0=x0,
            bounds=bounds,
            verbose=self.verbose,
            x_scale="jac",
        )
        return model

    def _get_opt_func(self, data, model) -> Callable:
        data_periodogram = self.periodogram(data)

        def opt_func(x):
            # convert to gpu if necessary
            x = xp.asarray(x)
            model.update_free_parameters(x)
            model_ep = self.expected_periodogram(model)
            ratio = (data_periodogram / model_ep).flatten()
            residuals = ratio - xp.ones_like(ratio)
            return xp.to_cpu(residuals)

        return opt_func
