import numpy
from debiased_spatial_whittle.models.base import CovarianceModel, CompoundModel, ModelParameter
from debiased_spatial_whittle.backend import BackendManager


xp = BackendManager.get_backend()


class BivariateUniformCorrelation(CompoundModel):
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

    r = ModelParameter(default=0.0, bounds=(-0.99, 0.99), doc="Correlation")
    f = ModelParameter(default=1.0, bounds=(1e-2, 1e2), doc="Amplitude ratio")

    def __init__(self, base_model: CovarianceModel, *args, **kwargs):
        super(BivariateUniformCorrelation, self).__init__(
            [
                base_model,
            ],
            *args,
            **kwargs,
        )

    @property
    def base_model(self):
        return self.children[0]

    @base_model.setter
    def base_model(self, model):
        raise AttributeError("Base model cannot be set")

    def _compute(self, lags: xp.ndarray):
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
        acv11 = self.base_model._compute(lags)
        fill_in = xp.ones_like(self.r * self.f)
        column1 = xp.stack((acv11 * fill_in, acv11 * self.r * self.f), -1)
        column2 = xp.stack((acv11 * self.r * self.f, acv11 * fill_in * self.f ** 2), -1)
        return xp.stack((column1, column2), -1)

    def _gradient(self, x: xp.ndarray):
        """

        Parameters
        ----------
        x
            shape (ndim, m1, ..., mk)

        Returns
        -------
        gradient
            shape (m1, ..., mk, 2, 2, p + 2)
            where p is the number of parameters of the base model.
        """
        acv_base_model = self.base_model(x)
        gradient_base_model = self.base_model._gradient(x)
        # derivative w.r.t. r
        d_r = xp.zeros(acv_base_model.shape + (2, 2))
        d_r[..., 0, 1] = acv_base_model * self.f
        d_r[..., 1, 0] = acv_base_model * self.f
        # derivative w.r.t. f
        d_f = xp.zeros(acv_base_model.shape + (2, 2))
        d_f[..., 1, 1] = 2 * self.f * acv_base_model
        d_f[..., 0, 1] = acv_base_model * self.r
        d_f[..., 1, 0] = acv_base_model * self.r
        return dict(r=d_r, f=d_f)