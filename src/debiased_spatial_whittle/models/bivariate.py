import numpy
from debiased_spatial_whittle.models.base import CovarianceModel, ModelParameter
from debiased_spatial_whittle.backend import BackendManager


xp = BackendManager.get_backend()


class BivariateUniformCorrelation(CovarianceModel):
    """
    This class defines the simple case of a bivariate covariance model where a given univariate covariance model is
    used in parallel to a uniform correlation parameter.

    Attributes
    ----------
    base_model: CovarianceModel
        Base univariate covariance model

    r: Parameter
        Correlation parameter, float between -1 and 1

    f: Parameter
        Amplitude ratio, float, positive

    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
    >>> base_model = ExponentialModel(rho=12.)
    >>> bivariate_model = BivariateUniformCorrelation(base_model, r=0.3, f=2.)
    """

    r = ModelParameter(default=0.0, bounds=(-0.99, 0.99), doc="Correlation", latex_display="r")
    f = ModelParameter(default=1.0, bounds=(1e-2, 1e2), doc="Amplitude ratio", latex_display="f")

    def __init__(self, base_model: CovarianceModel, r=None, f=None, name=None):
        super().__init__((base_model,), r, f, name=name)

    @property
    def base_model(self):
        return self.children[0]

    @base_model.setter
    def base_model(self, model):
        raise AttributeError("Base model cannot be set")

    def compute(self, lags: xp.ndarray, r: xp.ndarray, f: xp.ndarray, *params) -> xp.ndarray:
        """
        Evaluates the covariance model at the passed lags. Since the model is bivariate,
        the returned array has two extra dimensions compared to the array lags, both of size
        two.

        Parameters
        ----------
        lags: ndarray
            lag array with shape (ndim, m1, m2, ..., mk)

        Returns
        -------
            Covariance values with shape (m1, m2, ..., mk, 2, 2)

        """
        child_params = params
        acv11 = self.base_model.compute(lags, *child_params)
        fill_in = xp.ones_like(r * f)
        column1 = xp.stack((acv11 * fill_in, acv11 * r * f), -1)
        column2 = xp.stack((acv11 * r * f, acv11 * fill_in * f ** 2), -1)
        return xp.stack((column1, column2), -1)