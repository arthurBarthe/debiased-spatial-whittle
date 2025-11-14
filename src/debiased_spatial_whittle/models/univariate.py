import numpy
from debiased_spatial_whittle.models.base import CovarianceModel, CompoundModel, ModelParameter
from debiased_spatial_whittle.backend import BackendManager


np = BackendManager.get_backend()


class ExponentialModel(CovarianceModel):
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
    >>> model(np.array([[0., 1.], [0., 0.]]))
    array([1.9881    , 1.62771861])
    >>> model.rho = 3
    >>> model.rho
    3
    >>> model.param.rho.bounds
    (0, inf)
    >>> model.free_parameters
    ['rho', 'sigma']
    >>> model.fix_parameter("rho")
    >>> model.free_parameters
    ['sigma']
    """

    rho = ModelParameter(default=1.0, bounds=(0, numpy.inf), doc="Range parameter")
    sigma = ModelParameter(
        default=1.0, bounds=(0, numpy.inf), doc="Amplitude parameter"
    )

    def _compute(self, lags: np.ndarray):
        d = np.sqrt(np.sum(lags**2, 0)) / self.rho
        return self.sigma**2 * np.exp(-d)

    def _gradient(self, lags: np.ndarray):
        d = np.sqrt(sum((lag**2 for lag in lags)))
        d_rho = (self.sigma / self.rho) ** 2 * d * np.exp(-d / self.rho)
        d_sigma = 2 * self.sigma * np.exp(-d / self.rho)
        return dict(rho=d_rho, sigma=d_sigma)


class SquaredExponentialModel(CovarianceModel):
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
    >>> model(np.array([[0., 1.], [0., 0.]]))
    array([1.9881    , 1.94873298])
    """

    rho = ModelParameter(default=1.0, bounds=(0, np.inf), doc="Range parameter")
    sigma = ModelParameter(default=1.0, bounds=(0, np.inf), doc="Amplitude parameter")

    def _compute(self, lags: np.ndarray):
        d = np.sum(lags**2, 0) / (2 * self.rho**2)
        return self.sigma**2 * np.exp(-d)

    def _gradient(self, lags: np.ndarray):
        """
        Provides the derivatives of the covariance model evaluated at the passed lags with respect to
        the model's parameters.

        Examples
        --------
        >>> model = SquaredExponentialModel(rho=2, sigma=1.41)
        >>> model.gradient(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), [model.param.rho, model.param.sigma])
        array([[0.        , 2.82      ],
               [0.21931151, 2.48864127],
               [0.21931151, 2.48864127],
               [0.38708346, 2.19621821]])
        """
        d2 = sum((lag**2 for lag in lags))
        d_rho = (
            self.rho ** (-3) * d2 * self.sigma**2 * np.exp(-1 / 2 * d2 / self.rho**2)
        )
        d_sigma = 2 * self.sigma * np.exp(-1 / 2 * d2 / self.rho**2)
        return dict(rho=d_rho, sigma=d_sigma)


class Matern32Model(CovarianceModel):
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

    rho = ModelParameter(default=1.0, bounds=(0, np.inf))
    sigma = ModelParameter(default=1.0, bounds=(0, np.inf))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute(self, lags: np.ndarray):
        d = np.sqrt(np.sum(lags**2, 0))
        return (
            self.sigma**2
            * (1 + np.sqrt(3) * d / self.rho)
            * np.exp(-np.sqrt(3) * d / self.rho)
        )

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()


class Matern52Model(CovarianceModel):
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

    rho = ModelParameter(default=1.0, bounds=(0, np.inf))
    sigma = ModelParameter(default=1.0, bounds=(0, np.inf))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute(self, lags: np.ndarray):
        d = np.sqrt(np.sum(lags**2, 0))
        temp = np.sqrt(5) * d / self.rho
        return self.sigma**2 * (1 + temp + temp**2 / 3) * np.exp(-temp)

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()


class RationalQuadraticModel(CovarianceModel):
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

    rho = ModelParameter(default=1.0, bounds=(0.0, np.inf))
    alpha = ModelParameter(default=1.0, bounds=(0, np.inf))
    sigma = ModelParameter(default=1.0, bounds=(0, np.inf))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute(self, lags: np.array):
        d2 = np.sum(lags**2, 0) / (2 * self.rho**2)
        return self.sigma**2 * np.power(1 + d2 / self.alpha, -self.alpha)

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()


class NuggetModel(CompoundModel):
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
    >>> model(np.array([[0., 1., 2.]]))
    array([1.        , 0.9965338 , 0.98620712])
    >>> model = NuggetModel(model, nugget=0.1)
    >>> model(np.array([[0., 1., 2.]]))
    array([1.        , 0.89688042, 0.88758641])
    """

    nugget = ModelParameter(default=0.0, bounds=(0, 1), doc="Nugget amplitude")

    def __init__(self, model, *args, **kwargs):
        super().__init__(
            [
                model,
            ],
            *args,
            **kwargs,
        )

    def _compute(self, lags: np.ndarray):
        n_spatial_dim = lags.shape[0]
        zero_lag = np.zeros((n_spatial_dim, lags.shape[-1]))
        variance = self.children[0]._compute(zero_lag)
        return np.all(lags == 0, 0) * self.nugget * variance + (
            1 - self.nugget
        ) * self.children[0]._compute(lags)


class AnisotropicModel(CompoundModel):
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
    >>> model = AnisotropicModel(base_model, eta=1.5, phi=np.pi / 3)
    """

    eta = ModelParameter(default=1, bounds=(0, np.inf))
    phi = ModelParameter(default=0, bounds=(-np.pi / 2, np.pi / 2))

    def __init__(self, base_model: CovarianceModel, *args, **kwargs):
        super().__init__(
            [
                base_model,
            ],
            *args,
            **kwargs,
        )

    @property
    def scaling_matrix(self):
        return np.array([[self.eta, 0], [0, 1 / self.eta]])

    @property
    def rotation_matrix(self):
        return np.array(
            [
                [np.cos(self.phi), -np.sin(self.phi)],
                [np.sin(self.phi), np.cos(self.phi)],
            ]
        )

    def _compute(self, lags: np.ndarray):
        lags = np.swapaxes(lags, 0, -1)
        lags = np.expand_dims(lags, -1)
        lags = np.matmul(self.rotation_matrix, lags)
        lags = np.matmul(self.scaling_matrix, lags)
        lags = np.squeeze(lags, -1)
        lags = np.swapaxes(lags, 0, -1)
        return self.children[0]._compute(lags)