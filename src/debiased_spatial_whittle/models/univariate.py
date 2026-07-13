import numpy
from debiased_spatial_whittle.models.base import CovarianceModel, BaseCovarianceModel, ModelParameter
from debiased_spatial_whittle.backend import BackendManager


xp = BackendManager.get_backend()


class ExponentialModel(BaseCovarianceModel):
    """
    Implements the Exponential covariance model.

    Attributes
    ----------
    rho: ModelParameter
        length scale parameter

    sigma: ModelParameter
        amplitude parameter

    Examples
    --------
    >>> model = ExponentialModel(rho=5, sigma=1.41)
    >>> model(xp.array([[0., 1.], [0., 0.]]))
    array([1.9881    , 1.62771861])
    >>> model.rho = 3
    >>> model.rho
    3
    >>> model.rho.bounds
    (0, inf)
    >>> model.free_parameter_names
    ('ExponentialModel_rho', 'ExponentialModel_sigma')
    """

    rho = ModelParameter(default=1.0, bounds=(0, numpy.inf), doc="Range parameter", latex_display=r"\rho")
    sigma = ModelParameter(
        default=1.0, bounds=(0, numpy.inf), doc="Amplitude parameter", latex_display=r"\sigma"
    )

    def __init__(self, rho=None, sigma=None, name=None):
        super().__init__(rho, sigma, name=name)

    def compute(self, lags: xp.ndarray, rho: xp.ndarray, sigma: xp.ndarray) -> xp.ndarray:
        d = xp.sqrt(xp.sum(lags ** 2, 0)) / rho
        return sigma**2 * xp.exp(-d)


class SquaredExponentialModel(BaseCovarianceModel):
    """
    Implements the Squared Exponential covariance model, or Gaussian covariance model.

    Attributes
    ----------
    rho: ModelParameter
        length scale parameter

    sigma: ModelParameter
        amplitude parameter

    Examples
    --------
    >>> model = SquaredExponentialModel(rho=5, sigma=1.41)
    >>> model(xp.array([[0., 1.], [0., 0.]]))
    array([1.9881    , 1.94873298])
    """

    rho = ModelParameter(default=1.0, bounds=(0, xp.inf), doc="Range parameter", latex_display=r"\rho")
    sigma = ModelParameter(default=1.0, bounds=(0, xp.inf), doc="Amplitude parameter", latex_display=r"\sigma")

    def __init__(self, rho=None, sigma=None, name=None):
        super().__init__(rho, sigma, name=name)

    def compute(self, lags: xp.ndarray, rho: xp.ndarray, sigma: xp.ndarray) -> xp.ndarray:
        d = xp.sum(lags ** 2, 0) / (2 * rho ** 2)
        return sigma**2 * xp.exp(-d)


class Matern32Model(BaseCovarianceModel):
    """
    Implements the Matern Covariance kernel with slope parameter 3/2.

    Attributes
    ----------
    rho: ModelParameter
        length scale parameter of the kernel

    sigma: ModelParameter
        amplitude parameter of the kernel

    Examples
    --------
    >>> model = Matern32Model(rho=5, sigma=1)
    """

    rho = ModelParameter(default=1.0, bounds=(0, xp.inf))
    sigma = ModelParameter(default=1.0, bounds=(0, xp.inf))

    def __init__(self, rho=None, sigma=None, name=None):
        super().__init__(rho, sigma, name=name)

    def compute(self, lags: xp.ndarray, rho: xp.ndarray, sigma: xp.ndarray) -> xp.ndarray:
        d = xp.sqrt(xp.sum(lags ** 2, 0))
        sqrt3 = numpy.sqrt(3)
        return (
                sigma ** 2
                * (1 + sqrt3 * d / rho)
                * xp.exp(-sqrt3 * d / rho)
        )


class Matern52Model(BaseCovarianceModel):
    """
    Implements the Matern Covariance kernel with slope parameter 5/2.

    Attributes
    ----------
    rho: ModelParameter
        length scale parameter of the kernel

    sigma: ModelParameter
        amplitude parameter of the kernel

    Examples
    --------
    >>> model = Matern52Model(rho=10)
    >>> model = Matern52Model(rho=10, sigma=0.9)
    """

    rho = ModelParameter(default=1.0, bounds=(0, xp.inf))
    sigma = ModelParameter(default=1.0, bounds=(0, xp.inf))

    def __init__(self, rho=None, sigma=None, name=None):
        super().__init__(rho, sigma, name=name)

    def compute(self, lags: xp.ndarray, rho: xp.ndarray, sigma: xp.ndarray) -> xp.ndarray:
        d = xp.sqrt(xp.sum(lags ** 2, 0))
        temp = numpy.sqrt(5) * d / rho
        return sigma**2 * (1 + temp + temp**2 / 3) * xp.exp(-temp)


class RationalQuadraticModel(BaseCovarianceModel):
    """
    Implements the Rational Quadratic Covariance Kernel.

    Attributes
    ----------
    rho: ModelParameter
        length scale parameter of the kernel

    alpha: ModelParameter
        alpha parameter of the kernel

    sigma: ModelParameter
        amplitude parameter of the kernel

    Examples
    --------
    >>> model = RationalQuadraticModel(rho=20, alpha=1.5)
    """

    rho = ModelParameter(default=1.0, bounds=(0.0, xp.inf))
    alpha = ModelParameter(default=1.0, bounds=(0, xp.inf))
    sigma = ModelParameter(default=1.0, bounds=(0, xp.inf))

    def __init__(self, rho=None, alpha=None, sigma=None, name=None):
        super().__init__(rho, alpha, sigma, name=name)

    def compute(self, lags: xp.array, rho: xp.ndarray, alpha: xp.ndarray, sigma: xp.ndarray) -> xp.ndarray:
        d2 = xp.sum(lags ** 2, 0) / (2 * rho ** 2)
        return sigma**2 * (1 + d2 / alpha) ** (-alpha)


class NuggetModel(CovarianceModel):
    """
    Allows to add a nugget to a base covariance model. The nugget parameter is between 0 and 1 and characterises the
    proportion of the variance due to the nugget. For instance, if the base model has variance 2, using a Nugget model
    on top with nugget parameter 0.1 will result in a model whose variance is still 2, but with a nugget of 0.2.

    Properties
    ----------
    nugget: ModelParameter
        Proportion of variance explained by the nugget

    Examples
    --------
    >>> model = SquaredExponentialModel(rho=12, sigma=1)
    >>> model(xp.array([[0., 1., 2.]]))
    array([1.        , 0.9965338 , 0.98620712])
    >>> model = NuggetModel(model, nugget=0.1)
    >>> model(xp.array([[0., 1., 2.]]))
    array([1.        , 0.89688042, 0.88758641])
    """

    nugget = ModelParameter(default=0.0, bounds=(0, 1), doc="Nugget amplitude", latex_display=r"\delta")

    def __init__(self, base_model, nugget=None, name=None):
        super().__init__((base_model,), nugget, name=name)

    @property
    def base_model(self):
        return self.children[0]

    def compute(self, lags: xp.ndarray, nugget: xp.ndarray, *params) -> xp.ndarray:
        child_params = params
        n_spatial_dim = lags.shape[0]
        zero_lag = xp.zeros((n_spatial_dim, ))
        variance = self.children[0].compute(zero_lag, *child_params)
        return xp.all(lags == 0, 0) * nugget * variance + (
            1 - nugget
        ) * self.children[0].compute(lags, *child_params)


class AnisotropicModel(CovarianceModel):
    """
    Allows to define an anisotropic model based on a base isotropic model via a scaling + rotation transform.
    Dimension 2.

    Attributes
    ----------
    base_model: ModelInterface
        Underlying covariance modelCovariance model

    eta: ModelParameter
        Scaling factor

    phi: ModelParameter
        Rotation angle

    Examples
    --------
    >>> base_model = SquaredExponentialModel(rho=10)
    >>> model = AnisotropicModel(base_model, eta=1.5, phi=xp.pi / 3)
    """

    eta = ModelParameter(default=1, bounds=(0, xp.inf))
    phi = ModelParameter(default=0, bounds=(-xp.pi / 2, xp.pi / 2))

    def __init__(self, base_model, eta=None, phi=None, name=None):
        super().__init__((base_model,), eta, phi, name=name)

    @property
    def base_model(self):
        return self.children[0]

    @property
    def scaling_matrix(self):
        return xp.array([[self.eta, 0], [0, 1 / self.eta]])

    @property
    def rotation_matrix(self):
        return xp.array(
            [
                [xp.cos(self.phi), -xp.sin(self.phi)],
                [xp.sin(self.phi), xp.cos(self.phi)],
            ]
        )

    def compute(self, lags: xp.ndarray, eta: xp.ndarray, phi: xp.ndarray, *params) -> xp.ndarray:
        child_params = params
        lags = xp.swapaxes(lags, 0, -1)
        lags = xp.expand_dims(lags, -1)
        lags = xp.matmul(self.rotation_matrix, lags)
        lags = xp.matmul(self.scaling_matrix, lags)
        lags = xp.squeeze(lags, -1)
        lags = xp.swapaxes(lags, 0, -1)
        return self.children[0].compute(lags, *child_params)


class AmplitudeModel(CovarianceModel):
    """
    A model that applies an amplitude scaling to a base covariance model.
    The covariance is scaled by sigma^2.
    
    Attributes
    ----------
    sigma : ModelParameter
        Amplitude scaling parameter
    
    base_model : CovarianceModel
        The base covariance model to scale
    
    Examples
    --------
    >>> base_model = ExponentialModel(rho=5)
    >>> model = AmplitudeModel(base_model, sigma=2.0)
    """
    
    sigma = ModelParameter(default=1.0, bounds=(0, xp.inf), doc="Amplitude parameter")
    
    def __init__(self, base_model, sigma=None, name=None):
        super().__init__((base_model,), sigma, name=name)
    
    @property
    def base_model(self):
        return self.children[0]
    
    def compute(self, lags: xp.ndarray, sigma: xp.ndarray, *params) -> xp.ndarray:
        return sigma**2 * self.children[0].compute(lags, *params)