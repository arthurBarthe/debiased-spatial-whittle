from typing import Callable

from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.samples import SampleOnRectangularGrid
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
        self.periodogram = periodogram
        self.expected_periodogram = expected_periodogram
        self.verbose = verbose

    def __call__(self, data: np.array, model: CovarianceModel):
        x0 = model.free_parameter_values_to_array_deep()
        bounds = np.array(list(zip(*model.free_parameter_bounds_to_list_deep())))

        # convert to cpu
        x0 = np.to_cpu(x0)
        bounds = np.to_cpu(bounds)

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
            x = np.asarray(x)
            model.update_free_parameters(x)
            model_ep = self.expected_periodogram(model)
            ratio = (data_periodogram / model_ep).flatten()
            residuals = ratio - np.ones_like(ratio)
            return np.to_cpu(residuals)

        return opt_func
