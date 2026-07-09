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


class NuggetModel(CovarianceModel):
    """
    Bivariate version that allows to add a nugget to a base covariance model.
    Each variate has its own nugget parameter.
    
    Properties
    ----------
    nugget_0 : ModelParameter
        Proportion of variance explained by the nugget for variate 0
    nugget_1 : ModelParameter
        Proportion of variance explained by the nugget for variate 1
    
    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
    >>> from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
    >>> base_model = ExponentialModel(rho=25.)
    >>> bivariate_model = BivariateUniformCorrelation(base_model, r=0.2, f=1.)
    >>> model = NuggetModel(bivariate_model, nugget0=0.1, nugget1=0.2)
    >>> from debiased_spatial_whittle.backend import BackendManager
    >>> xp = BackendManager.get_backend()
    >>> print(model)
    >>> model(xp.array([[0., 1., 2.], [0., 0., 0.]]))
    """

    nugget0 = ModelParameter(default=0.0, bounds=(0, 1), doc="Nugget amplitude variate 0", latex_display=r"\delta_0")
    nugget1 = ModelParameter(default=0.0, bounds=(0, 1), doc="Nugget amplitude variate 1", latex_display=r"\delta_1")

    def __init__(self, base_model, nugget0=None, nugget1=None, name=None):
        super().__init__((base_model,), nugget0, nugget1, name=name)

    @property
    def base_model(self):
        return self.children[0]

    def compute(self, lags: xp.ndarray, nugget_0: xp.ndarray, nugget_1: xp.ndarray, *params) -> xp.ndarray:
        child_params = params
        n_spatial_dim = lags.shape[0]
        zero_lag = xp.zeros((n_spatial_dim, lags.shape[-1]))
        
        # Compute base model covariance
        cov = self.children[0].compute(lags, *child_params)
        
        # Add nugget to diagonal for each variate
        # cov shape: (..., 2, 2)
        # We need to add nugget_i * variance at zero lag to the (i, i) element
        variance_0 = self.children[0].compute(zero_lag, *child_params)[..., 0, 0]
        variance_1 = self.children[0].compute(zero_lag, *child_params)[..., 1, 1]
        
        # Create nugget matrix
        nugget_matrix = xp.zeros_like(cov)
        nugget_matrix[..., 0, 0] = xp.all(lags == 0, 0) * nugget_0 * variance_0
        nugget_matrix[..., 1, 1] = xp.all(lags == 0, 0) * nugget_1 * variance_1

        # matrix applied to base covariance
        mat = xp.array([[1 - nugget_0, (1 - nugget_0) * (1 - nugget_1)],
                        [(1 - nugget_0) * (1 - nugget_1), 1 - nugget_1]])
        
        # Apply nugget and scale
        return mat * cov + nugget_matrix


class AmplitudeModel(CovarianceModel):
    """
    Bivariate version that applies amplitude scaling to a base covariance model.
    Each variate has its own sigma parameter.
    
    Properties
    ----------
    sigma_0 : ModelParameter
        Amplitude scaling parameter for variate 0
    sigma_1 : ModelParameter
        Amplitude scaling parameter for variate 1
    
    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
    >>> from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
    >>> base_model = ExponentialModel(rho=12.)
    >>> bivariate_model = BivariateUniformCorrelation(base_model, r=0.3, f=2.)
    >>> model = AmplitudeModel(bivariate_model, sigma_0=2.0, sigma_1=1.5)
    """

    sigma_0 = ModelParameter(default=1.0, bounds=(0, xp.inf), doc="Amplitude parameter variate 0")
    sigma_1 = ModelParameter(default=1.0, bounds=(0, xp.inf), doc="Amplitude parameter variate 1")

    def __init__(self, base_model, sigma_0=None, sigma_1=None, name=None):
        super().__init__((base_model,), sigma_0, sigma_1, name=name)

    @property
    def base_model(self):
        return self.children[0]

    def compute(self, lags: xp.ndarray, sigma_0: xp.ndarray, sigma_1: xp.ndarray, *params) -> xp.ndarray:
        cov = self.children[0].compute(lags, *params)
        # Scale each variate by its sigma
        # cov shape: (..., 2, 2)
        scaling = xp.array([[sigma_0**2, 0], [0, sigma_1**2]])
        # Broadcast scaling to match cov shape
        # scaling needs to be broadcast to the first dimensions of cov
        scaling_expanded = xp.reshape(scaling, (1,) * (cov.ndim - 2) + (2, 2))
        return cov * scaling_expanded