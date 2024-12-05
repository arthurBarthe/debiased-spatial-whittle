try:
    import torch
except ModuleNotFoundError:
    pass

from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()
fftn, ifftn = BackendManager.get_fft_methods()

import numpy

# In this file we define covariance models
from abc import ABC, abstractmethod

from typing import Tuple, List, Dict, Union


class Parameter:
    """
    Class designed to handle parameters of model. Note that a Parameter can have it own core value, or might just
    be a pointer to another parameter. In particular, two parameters might point to a single other parameter.
    In that case, those two parameters always have the same value.
    """

    def __init__(self, name: str, bounds: Tuple[float, float]):
        # TODO hide point_to
        # TODO add registry to ensure no duplicate names? Add access to registry by name of parameter?
        self.point_to = None
        self.name = name
        self.bounds = bounds
        self.value = None
        self.init_guess = 0.9
        # fixme log scale not expected to work with gradients right now
        # might need a whole new class LogParameter
        self.log_scale = False

    # TODO add property name and make it point to if adequate

    @property
    def value(self):
        if self.point_to is not None:
            return self.point_to.value
        if self._value is None:
            return None
        if self.log_scale:
            return numpy.exp(self._value)
        return self._value

    @value.setter
    def value(self, v: float):
        if self.point_to is not None:
            self.point_to.value = v
        self._value = v

    @property
    def free(self):
        return self.value is None

    def merge_with(self, other):
        assert (
            self.free and other.free
        ), "Only free parameters (value set to None) can be merged together."
        new_name = self.name + " and " + other.name
        new_min = max(self.bounds[0], other.bounds[0])
        new_max = min(self.bounds[1], other.bounds[1])
        new_bounds = (new_min, new_max)
        new_parameter = Parameter(new_name, new_bounds)
        self.point_to = new_parameter
        other.point_to = new_parameter
        return new_parameter

    def __pow__(self, power, modulo=None):
        return self.value**power

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return self.value * other

    def __invert__(self):
        return 1 / self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __repr__(self):
        return f"{self.name}: {self.value} ... {self.bounds}"


class Parameters:
    """Wrapper for a dictionary of parameters. The keys are the names of the parameters in the model, but two
    different keys could point to a single Parameter object"""

    def __init__(self, parameters: List[Parameter]):
        # TODO make param_dict hidden
        self.param_dict = dict([(p.name, p) for p in parameters])

    @property
    def names(self):
        return list(self.param_dict.keys())

    @property
    def values(self):
        return [p.value for p in self]

    @property
    def bounds(self):
        return [p.bounds for p in self]

    @property
    def init_guesses(self):
        return [p.init_guess for p in self]

    def free_params(self):
        list_free = list(filter(lambda p: p.free, self))
        return Parameters(list_free)

    def __getitem__(self, item):
        return self.param_dict[item]

    # TODO add some checks that no two parameters have the same name

    def __setitem__(self, key, value):
        self.param_dict[key] = value

    def __iter__(self):
        for p in self.param_dict.values():
            yield p

    def __len__(self):
        return len(self.param_dict.keys())

    def update_values(self, updates: Dict[str, float]):
        for p_name, value in updates.items():
            self[p_name].value = value

    def __repr__(self):
        return "\n".join([p.__repr__() for p in self])


class ParametersUnion(Parameters):
    """A class to represent the union of parameters when using a combination of models
    to define a new model. Handles the case where several of the models are using a common parameter."""

    def __init__(self, parameters: List[Parameters]):
        param_list = []
        for i, params in enumerate(parameters):
            for p in params:
                p.name = p.name + "_" + str(i)
                param_list.append(p)
        super(ParametersUnion, self).__init__(param_list)


class CovarianceModel(ABC):
    """Abstract class for the definition of a covariance model."""

    def __init__(self, parameters: Parameters):
        self.params = parameters

    @property
    def param_bounds(self):
        return self.params.bounds

    @property
    def n_params(self) -> int:
        """Number of parameters of the model"""
        return len(self.params)

    @property
    def param_names(self) -> list[str]:
        """Names of the parameters of the model"""
        return self.params.names

    @property
    def param_values(self) -> list:
        """Values of the model's parameters"""
        return self.params.values

    @property
    def free_params(self) -> Parameters:
        """Free parameters of the model"""
        return self.params.free_params()

    @property
    def free_param_bounds(self):
        return self.free_params.bounds

    @property
    def has_free_params(self) -> bool:
        """Whether the model has any free parameter"""
        return len(self.free_params) > 0

    def update_free_params(self, updates):
        self.params.update(updates)

    def merge_parameters(self, param_names: Tuple[str, str]):
        """
        Merges two parameters into one. This can only be done for two free parameters
        """
        p1_name, p2_name = param_names
        p1 = self.params[p1_name]
        p2 = self.params[p2_name]
        new_param = p1.merge_with(p2)
        self.params[p1_name] = new_param
        self.params[p2_name] = new_param

    @property
    def name(self):
        """name of the model used for user outputs"""
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, x: np.ndarray):
        """
        Parameters
        ----------
        x: ndarray
            Array of spatial lags. The first dimension is used to index the dimensions of the domain.

        Returns
        -------
        cov: ndarray
            Covariance model evaluated at the passed lags

        Notes
        -----
        The first dimension of the passed array should correspond to the dimension of the space over which
        the random field is defined. Other dimensions can have any size.

        Examples
        --------
        >>> model = ExponentialModel()
        >>> model.rho = 12
        >>> model.sigma = 1
        >>> model(np.array([[0., 0., 0.], [0., 1., 2.]]))
        array([1.        , 0.92004441, 0.84648172])
        """
        raise NotImplementedError()

    def cov_mat_x1_x2(self, x1: np.ndarray, x2: np.ndarray = None) -> np.ndarray:
        """
        Compute the covariance matrix between between points in x1 and points in x2.

        Parameters
        ----------
        x1
            shape (N1, d), first set of locations
        x2
            shape (N2, d), second set of locations

        Returns
        -------
        covmat
            shape (N1, N2), covariance matrix
        """
        if x2 is None:
            x2 = x1
        x1 = np.expand_dims(x1, axis=1)
        x2 = np.expand_dims(x2, axis=0)
        lags = x1 - x2
        lags = np.transpose(lags, (2, 0, 1))
        return self(lags)

    def gradient(self, x: np.ndarray, params: Parameters):
        """Provides the gradient of the covariance functions at the passed lags with respect to
        the passed parameters"""
        gradient = dict([(p.name, 0) for p in params.param_dict.values()])
        g = self._gradient(x)
        for i, p in enumerate(self.params):
            if p in params.param_dict.values():
                # gradient[p.name] += np.take(g, i, axis=-1)
                gradient[p.name] += g[..., i]
        return gradient

    @abstractmethod
    def _gradient(self, x: np.ndarray):
        """Provide the gradient with respect to all parameters of the covariance model for the passed lags.
        The last dimension of the returned array should correspond to the parameters of the model."""
        pass

    def __setattr__(self, key: str, value: Union[float, Parameter]):
        if "params" in self.__dict__:
            if key in self.param_names:
                if isinstance(value, Parameter):
                    self.params[key] = value
                else:
                    self.params[key].value = value
                return
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.param_names:
            return self.params[item]
        else:
            raise AttributeError()

    def __repr__(self):
        return self.name + "(\n" + self.params.__repr__() + "\n)"

    def __getstate__(self):
        """
        Necessary for pickling and unpiclking

        Returns
        -------
        State as a dictionary
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Necessary for pickling and unpickling.
        Parameters
        ----------
        state
            Dictionary with the new state
        """
        self.__dict__.update(state)

    def __add__(self, other):
        """
        Add two covariance models, resulting in a covariance model whose covariance function is the
        sum of the two covariance functions.

        Parameters
        ----------
        other
            Another covariance model

        Returns
        -------
        new_model
            New covariance model
        """
        return SumCovarianceModel(self, other)


class SumCovarianceModel(CovarianceModel):
    """
    Defines a covariance model as the sum of two covariance models.
    """

    def __init__(self, model1, model2):
        """
        Parameters
        ----------
        model1
            First covariance model
        model2
            Second covariance modelp
        """
        self.model1 = model1
        self.model2 = model2
        params = ParametersUnion([model1.params, model2.params])
        super(SumCovarianceModel, self).__init__(params)

    def __call__(self, lags: np.ndarray):
        return self.model1(lags) + self.model2(lags)

    def _gradient(self, x: np.ndarray):
        grad1 = self.model1._gradient(x)
        grad2 = self.model2._gradient(x)
        return np.stack((grad1, grad2))


class SeparableModelOld(CovarianceModel):
    """Class for a separable covariance model based on a list of covariance models"""

    # TODO only works in dimension 2 right now

    def __init__(self, models: List[CovarianceModel]):
        self.models = models
        parameters = ParametersUnion([m.params for m in models])
        super().__init__(parameters)

    def __call__(self, lags: np.ndarray):
        cov_seqs = [
            model_i(
                [
                    lags_i,
                ]
            )
            for model_i, lags_i in zip(self.models, lags)
        ]
        return cov_seqs[0] * cov_seqs[1]

    def _gradient(self, lags):
        covs = [
            model_i(
                [
                    lags_i,
                ]
            )
            for model_i, lags_i in zip(self.models, lags)
        ]
        gradients = [
            model_i._gradient(
                [
                    lags_i,
                ]
            )
            for model_i, lags_i in zip(self.models, lags)
        ]
        g1 = gradients[0] * np.expand_dims(covs[1], axis=-1)
        g2 = gradients[1] * np.expand_dims(covs[0], axis=-1)
        return np.concatenate((g1, g2), axis=-1)

    def __repr__(self):
        return (
            "SeparableModel(\n" + "\n".join([m.__repr__() for m in self.models]) + "\n)"
        )


class SeparableModel(CovarianceModel):
    """Class for a separable covariance model based on a list of covariance models"""

    # TODO only works in dimension 2 right now
    # TODO we should allow for a semi-separable model: for instance, time and space are separable, but 2d space is not

    def __init__(self, models: List[CovarianceModel], dims: List):
        self.models = models
        self.dims = dims
        assert len(models) == len(dims)
        parameters = ParametersUnion([m.params for m in models])
        super(SeparableModel, self).__init__(parameters)

    def __call__(self, lags: List[np.ndarray]):
        lags = np.stack(lags, 0)
        acvs = []
        for i in range(len(self.models)):
            acvs.append(self.models[i](np.take(lags, self.dims[i], 0)))
        return np.prod(np.stack(acvs, 0), 0)

    def _gradient(self, lags):
        raise NotImplementedError()
        # covs = [model_i(lags) for model_i in self.models]
        # gradients = [model_i._gradient(lags) for model_i in self.models]
        # g1 = gradients[0] * np.expand_dims(covs[1], axis=-1)
        # g2 = gradients[1] * np.expand_dims(covs[0], axis=-1)
        # return np.concatenate((g1, g2), axis=-1)

    def __repr__(self):
        return (
            "SeparableModel(\n" + "\n".join([m.__repr__() for m in self.models]) + "\n)"
        )


class ExponentialModel(CovarianceModel):
    """
    Generic class for the definition of an exponential covariance model, in any number of dimensions.

    Attributes
    ----------
    rho: Parameter
        Lengthscale parameter

    sigma: Parameter
        Amplitude parameter

    nugget: Parameter
        Nugget parameter

    Examples
    --------
    >>> model = ExponentialModel()
    >>> model.rho = 12.
    >>> model.sigma = 1.
    """

    def __init__(self):
        sigma = Parameter("sigma", (1e-30, 1000))
        rho = Parameter("rho", (1e-30, 1000))
        nugget = Parameter("nugget", (1e-30, 1000))

        parameters = Parameters([rho, sigma, nugget])
        super(ExponentialModel, self).__init__(parameters)
        # set a default value to zero for the nugget
        self.nugget = 0.0

    def __call__(self, lags: np.ndarray):
        """

        Parameters
        ----------
        lags
            Array of lags

        Returns
        -------
        Exponential covariance function evaluated at the passed lags.

        Examples
        --------
        >>> model = ExponentialModel()
        >>> model.rho = 2
        >>> model.sigma = 1.41
        >>> model(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))
        array([1.9881    , 1.2058436 , 1.2058436 , 0.98026987])
        """
        d = np.sqrt(sum((lag**2 for lag in lags)))
        nugget_effect = self.nugget.value * np.all(lags == 0, axis=0)
        acf = self.sigma.value**2 * np.exp(-d / self.rho.value) + nugget_effect
        return acf

    def _gradient(self, lags: np.ndarray):
        """Provides the derivatives of the covariance model evaluated at the passed lags with respect to
        the model's parameters. The user should not call this method directly in general, instead they should
        use the gradient method.

        Examples
        --------
        >>> model = ExponentialModel()
        >>> model.rho = 2
        >>> model.sigma = 1.41
        >>> model.gradient(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), Parameters([model.rho, model.nugget]))
        {'rho': array([0.        , 0.3014609 , 0.3014609 , 0.34657773]), 'nugget': array([1., 0., 0., 0.])}
        """
        d = np.sqrt(sum((lag**2 for lag in lags)))
        d_rho = (
            (self.sigma.value / self.rho.value) ** 2 * d * np.exp(-d / self.rho.value)
        )
        d_sigma = 2 * self.sigma.value * np.exp(-d / self.rho.value)
        d_nugget = 1.0 * (d == 0)
        return np.stack((d_rho, d_sigma, d_nugget), axis=-1)

    def _gradient_reparamed(self, lags: np.ndarray):
        """Gradient when the parameters are log-transformed, i.e. rho,sigma,nugget = exp(param_values)"""
        rho, sigma, nugget = self.rho.value, self.sigma.value, self.nugget.value

        d = np.sqrt(sum((lag**2 for lag in lags)))
        d_rho = (sigma**2 / rho) * d * np.exp(-d / rho)
        d_sigma = 2 * sigma**2 * np.exp(-d / rho)
        d_nugget = 1 * (d == 0)
        return np.stack((d_rho, d_sigma, d_nugget), axis=-1)


class ExponentialModelUniDirectional(CovarianceModel):
    """Class for the implementation of a Unidirectional covariance model. At the moment, we only implement
    the case where the direction is aligned with one of the axis."""

    def __init__(self, axis: int):
        sigma = Parameter("sigma", (0.01, 1000))
        rho = Parameter("rho", (0.01, 1000))
        parameters = Parameters([rho, sigma])
        super(ExponentialModelUniDirectional, self).__init__(parameters)
        self.axis = axis
        # TODO add parameter for orientation

    def __call__(self, lags: List[np.ndarray]):
        d = np.abs(lags[self.axis])
        return self.sigma.value**2 * np.exp(-d / self.rho.value)

    def _gradient(self, lags: np.ndarray):
        d = np.abs(lags[self.axis])
        d_rho = (
            (self.sigma.value / self.rho.value) ** 2 * d * np.exp(-d / self.rho.value)
        )
        d_sigma = 2 * self.sigma.value * np.exp(-d / self.rho.value)
        return np.stack((d_rho, d_sigma), axis=-1)


from scipy.special import kv

try:
    from autograd.scipy.special import gamma, iv
    from autograd.extend import primitive, defvjp, defjvp
except ModuleNotFoundError:
    from scipy.special import gamma


def kv_(nu, z):
    if nu % 1 == 0:
        nu += 1e-6
    return (np.pi / 2) * (iv(-nu, z) - iv(nu, z)) / np.sin(nu * np.pi)


class MaternModel(CovarianceModel):
    #  TODO: needs more testing!
    def __init__(self):
        rho = Parameter("rho", (0.01, 1000))
        sigma = Parameter("sigma", (0.01, 1000))
        nu = Parameter("nu", (0.01, 1000))
        nugget = Parameter("nugget", (1e-30, 1000))

        parameters = Parameters([rho, sigma, nu, nugget])
        super(MaternModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        rho, sigma, v = self.rho.value, self.sigma.value, self.nu.value

        d = np.sqrt(sum((lag**2 for lag in lags)))
        mask = d == 0
        nugget_effect = sigma**2 + self.nugget.value

        # TODO: add specific cases for nu=1/2, 3/2, 5/2, inf
        const = 2 ** (1 - v) / gamma(v)
        args = np.sqrt(2 * v) * d / rho
        term2 = (args) ** v
        term3 = kv_(v, args + mask)  # TODO: bad solution
        acf = sigma**2 * const * term2 * term3
        acf += (nugget_effect - acf) * mask
        return acf  # sdfsadffasdf

    def f(
        self,
        freq_grid: Union[List, np.ndarray],
        infsum_grid: Union[List, np.ndarray],
        d: int = 2,
    ):
        """aliased spectral density, should match with the acf"""

        rho, sigma, v = self.rho.value, self.sigma.value, self.nu.value

        shape = freq_grid[0].shape
        d = len(shape)
        N = np.prod(shape)

        # TODO: wrong spectral density?
        args = (
            np.tile(infsum_grid[i], (N, 1, 1)) + freq_grid[i].reshape(N, 1, 1)
            for i in range(d)
        )
        omega2 = np.sum((arg**2 for arg in args))

        term1 = 2 ** (d) * np.pi ** (d / 2) * gamma(v + d / 2) * (2 * v) ** v
        term2 = 1 / (gamma(v) * rho ** (2 * v))
        term3 = (2 * v / rho**2 + 4 * np.pi**2 * omega2) ** (-v - d / 2)
        f = sigma**2 * term1 * term2 * term3

        return np.sum(f, axis=(1, 2)).reshape(shape) + self.nugget.value

    def _gradient(self):
        pass


class SquaredExponentialModel(CovarianceModel):
    """
    Generic class for the definition of a Squared Exponential Covariance model, in any number of dimensions.

    Attributes
    ----------
    rho: Parameter
        Lengthscale parameter

    sigma: Parameter
        Amplitude parameter

    nugget: Parameter
        Nugget parameter

    Examples
    --------
    >>> model = SquaredExponentialModel()
    >>> model.rho = 12.
    >>> model.sigma = 1.
    """

    def __init__(self):
        rho = Parameter("rho", (0.01, 1000))
        sigma = Parameter("sigma", (0.01, 1000))
        nugget = Parameter("nugget", (1e-30, 1000))

        parameters = Parameters([rho, sigma, nugget])
        super(SquaredExponentialModel, self).__init__(parameters)
        # set a default value to zero for the nugget
        self.nugget = 0.0

    def __call__(self, lags: np.ndarray):
        d2 = np.sum(lags**2, axis=0)
        nugget_effect = self.nugget.value * np.all(lags == 0, axis=0)
        acf = (
            self.sigma.value**2 * np.exp(-0.5 * d2 / self.rho.value**2) + nugget_effect
        )
        return acf

    def f(
        self,
        freq_grid: Union[list, np.ndarray],
        infsum_grid: Union[list, np.ndarray],
        d: int = 2,
    ):
        """aliased spectral density, should match with the acf"""

        shape = freq_grid[0].shape
        N = np.prod(shape)

        args = (
            np.tile(infsum_grid[i], (N, 1, 1)) + freq_grid[i].reshape(N, 1, 1)
            for i in range(d)
        )
        omega2 = np.sum((arg**2 for arg in args))

        # f = sigma2*(2*np.pi*rho**2)**(d/2)*np.exp(-2*(np.pi*rho)**2 * omega2) #+ nugget/(2*np.pi)**2
        f = (
            self.sigma.value**2
            * self.rho.value**2
            * (2 * np.pi) ** (d / 2)
            * np.exp(-0.5 * (self.rho.value**2 * omega2))
        )  # /(2*np.pi)**2
        return np.sum(f, axis=(1, 2)).reshape(shape) + self.nugget.value

    def _gradient(self, lags: np.ndarray):
        """Provides the derivatives of the covariance model evaluated at the passed lags with respect to
        the model's parameters.

        >>> model = SquaredExponentialModel()
        >>> model.rho = 2
        >>> model.sigma = 1.41
        >>> model.gradient(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), Parameters([model.rho, model.nugget]))
        {'rho': array([0.        , 0.21931151, 0.21931151, 0.38708346]), 'nugget': array([1., 0., 0., 0.])}
        """
        d2 = sum((lag**2 for lag in lags))
        d_rho = (
            self.rho.value ** (-3)
            * d2
            * self.sigma.value**2
            * np.exp(-1 / 2 * d2 / self.rho.value**2)
        )
        d_sigma = 2 * self.sigma.value * np.exp(-1 / 2 * d2 / self.rho.value**2)
        d_nugget = 1 * (d2 == 0)
        return np.stack((d_rho, d_sigma, d_nugget), axis=-1)

    def _gradient_reparamed(self, lags: np.ndarray):
        """Gradient when the parameters are log-transform, i.e. rho,sigma,nugget = exp(param_values)"""
        rho, sigma, nugget = self.rho.value, self.sigma.value, self.nugget.value

        d2 = sum((lag**2 for lag in lags))
        d_rho = (sigma / rho) ** 2 * d2 * np.exp(-0.5 * d2 / rho**2)
        d_sigma = 2 * sigma**2 * np.exp(-0.5 * d2 / rho**2)
        d_nugget = 1 * (d2 == 0)
        return np.stack((d_rho, d_sigma, d_nugget), axis=-1)


# TODO this should not be just a covariance model. Create more general class for model with parameters
class TMultivariateModel(CovarianceModel):
    """
    Model corresponding to a t-multivariate distribution.
    """

    def __init__(self, covariance_model: CovarianceModel):
        self.covariance_model = covariance_model
        nu = Parameter("nu", (1, 1000))
        parameters = ParametersUnion(
            [
                covariance_model.params,
                Parameters(
                    [
                        nu,
                    ]
                ),
            ]
        )
        super(TMultivariateModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        acv_gaussian = self.covariance_model(lags)
        return acv_gaussian * self.nu_1.value / (self.nu_1.value - 2)

    def _gradient(self, lags: np.ndarray):
        return (
            self.covariance_model._gradient(lags) * self.nu.value / (self.nu.value - 2)
        )


class SquaredModel(CovarianceModel):
    """
    Covariance model for a process defined pointwise as the square of a Gaussian process
    """

    def __init__(self, latent_model: CovarianceModel):
        self.latent_model = latent_model
        super(SquaredModel, self).__init__(self.latent_model.params)

    def __call__(self, lags: np.ndarray):
        return 2 * self.latent_model(lags) ** 2 + self.latent_model.sigma**2

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class ChiSquaredModel(CovarianceModel):
    """
    Covariance model for a Chi-Squared random field
    """

    def __init__(self, latent_model: CovarianceModel):
        self.latent_model = latent_model
        dof = Parameter("dof", (1, 1000))
        super(ChiSquaredModel, self).__init__(
            ParametersUnion(
                [
                    self.latent_model.params,
                    Parameters(
                        [
                            dof,
                        ]
                    ),
                ]
            )
        )

    def __call__(self, lags: np.ndarray):
        return (
            self.dof_1.value
            * (2 * self.latent_model(lags) ** 2 + self.latent_model.sigma**2)
            + self.dof_1.value * (self.dof_1.value - 1) * self.latent_model.sigma**2
        )

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class BivariateUniformCorrelation(CovarianceModel):
    """
    This class defines the simple case of a bivariate covariance model where a given univariate covariance model is
    used in parallel to a uniform correlation parameter.

    Attributes
    ----------
    base_model: CovarianceModel
        Base univariate covariance model

    r_0: Parameter
        Correlation parameter, float between -1 and 1

    f_0: Parameter
        Amplitude ratio, float, positive

    Examples
    --------
    >>> base_model = ExponentialModel()
    >>> base_model.rho = 12.
    >>> base_model.sigma = 1.
    >>> bivariate_model = BivariateUniformCorrelation(base_model)
    >>> bivariate_model.r_0 = 0.75
    >>> bivariate_model.f_0 = 2.3
    """

    def __init__(self, base_model: CovarianceModel):
        self.base_model = base_model
        r = Parameter("r", (-0.99, 0.99))
        f = Parameter("f", (0.1, 10))
        parameters = ParametersUnion([Parameters([r, f]), base_model.params])
        super(BivariateUniformCorrelation, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
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
            Covariance values with shape (ndim, m1, m2, ..., mk, 2, 2)

        """
        acv11 = self.base_model(lags)
        # TODO looks ugly that we use r_0. Reconsider implementation of parameters?
        out = np.zeros(acv11.shape + (2, 2))
        out = BackendManager.convert(out)
        out[..., 0, 0] = acv11
        out[..., 1, 1] = acv11 * self.f_0.value**2
        out[..., 0, 1] = acv11 * self.r_0.value * self.f_0.value
        out[..., 1, 0] = acv11 * self.r_0.value * self.f_0.value
        return out

    def _gradient(self, x: np.ndarray):
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
        acv11 = self.base_model(x)
        gradient_base_model = self.base_model._gradient(x)
        # gradient 11
        temp = np.stack((np.zeros_like(acv11), np.zeros_like(acv11)), axis=-1)
        gradient_11 = np.concatenate((temp, gradient_base_model), axis=-1)
        # gradient 12
        temp = np.stack((acv11 * self.f_0.value, acv11 * self.r_0.value), axis=-1)
        gradient_12 = np.concatenate(
            (temp, gradient_base_model * self.r_0.value * self.f_0.value), axis=-1
        )
        # gradient 21
        gradient_21 = gradient_12
        # gradient 22
        temp = np.stack((np.zeros_like(acv11), 2 * self.f_0.value * acv11), axis=-1)
        gradient_22 = np.concatenate(
            (temp, gradient_base_model * self.f_0.value**2), axis=-1
        )
        row1 = np.stack((gradient_11, gradient_12), axis=x.ndim - 1)
        row2 = np.stack((gradient_21, gradient_22), axis=x.ndim - 1)
        return np.stack((row1, row2), axis=x.ndim - 1)


class TransformedModel(CovarianceModel):
    """
    This class defines a covariance model obtained from a covariance model for some input random fields,
    to which a transform is applied in the spectral domain. For now this is specific to the biharmonic equation.
    """

    def __init__(self, input_model: CovarianceModel, transform_func):
        self.input_model = input_model
        self.transform_func = transform_func
        transform_param = Parameter("logD", (1, 50))
        eta_param = Parameter("eta", (-1, 1))
        parameters = ParametersUnion(
            [Parameters([transform_param, eta_param]), input_model.params]
        )
        super(TransformedModel, self).__init__(parameters)

    def transform_on_grid(self, ks):
        return self.transform_func(self.logD_0.value, self.eta_0.value, ks)

    def __call__(self):
        raise NotImplementedError()

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class NewTransformedModel(CovarianceModel):
    """
    Similar to the above class, but uses circulant embedding to alleviate some of the approximations.
    """

    def __init__(self, input_model: CovarianceModel, transform):
        self.input_model = input_model
        self.transform = transform
        transform_param = Parameter("logD", (1, 50))
        eta_param = Parameter("eta", (-1, 1))
        nu_param = Parameter("nu", (-np.pi / 2, np.pi / 2))
        z2_param = Parameter("logz2", (3, 5))
        parameters = ParametersUnion(
            [
                Parameters([transform_param, eta_param, nu_param, z2_param]),
                input_model.params,
            ]
        )
        super(NewTransformedModel, self).__init__(parameters)

    def transform_on_grid(self, ks):
        return self.transform(
            self.logD_0.value, self.eta_0.value, self.nu_0.value, self.logz2_0.value, ks
        )

    def call_on_rectangular_grid(self, grid):
        # periodic covariance of the input model
        acv = grid.autocov(self.input_model)
        # to spectral domain
        f = fftn(acv, axes=(0, 1))
        # apply the frequency-domain mapping
        transform = self.transform_on_grid(grid.fourier_frequencies2)
        if BackendManager.backend_name == "numpy":
            transform_transpose = np.transpose(transform, (0, 1, -1, -2))
        elif BackendManager.backend_name == "torch":
            transform_transpose = np.transpose(transform, -1, -2).to(
                dtype=torch.complex128
            )
            transform = transform.to(dtype=torch.complex128)
        term1 = np.matmul(f, transform_transpose)
        return ifftn(np.matmul(transform, term1), axes=(0, 1))

    def __call__(self, lags: np.ndarray):
        raise NotImplementedError(
            "The autocovariance of this model can only be evaluated in specific cases"
        )

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class NewTransformedModel2(CovarianceModel):
    """
    Similar to the above class, but uses circulant embedding to alleviate some of the approximations.
    """

    def __init__(
        self, input_model: CovarianceModel, transform_function, transform_params
    ):
        self.input_model = input_model
        self.transform = transform_function
        super(NewTransformedModel, self).__init__(transform_params)

    def transform_on_grid(self, ks):
        return self.transform(*self.params.values, ks)

    def call_on_rectangular_grid(self, grid):
        from numpy.fft import fftn, ifftn

        # periodic covariance of the input model
        acv = grid.autocov(self.input_model)
        # to spectral domain
        f = fftn(acv, axes=(0, 1))
        # apply the frequency-domain mapping
        transform = self.transform_on_grid(grid.fourier_frequencies2)
        transform_transpose = np.transpose(transform, (0, 1, -1, -2))
        term1 = np.matmul(f, transform_transpose)
        return ifftn(np.matmul(transform, term1), axes=(0, 1))

    def __call__(self, lags: np.ndarray):
        raise NotImplementedError(
            "The autocovariance of this model can only be evaluated in specific cases"
        )

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class MaternCovarianceModel(CovarianceModel):
    """
    Generic class for the definition of a Matern Covariance Model, in any number of dimensions.

    Attributes
    ----------
    rho: Parameter
        Lengthscale parameter

    sigma: Parameter
        Amplitude parameter

    nu: Parameter
        Slope parameter. Faster for values 0.5 and 1.5.

    Examples
    --------
    >>> model = MaternCovarianceModel()
    >>> model.rho = 12.
    >>> model.sigma = 1.
    >>> model.nu = 1.5
    """

    def __init__(self):
        rho = Parameter("rho", (0.01, 1000))
        sigma = Parameter("sigma", (0.01, 1000))
        nu = Parameter("nu", (0.01, 100))
        parameters = Parameters([rho, sigma, nu])
        super(MaternCovarianceModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        d = np.sqrt(np.sum(lags**2, axis=0))
        sigma, rho, nu = self.sigma.value, self.rho.value, self.nu.value
        if nu == 0.5:
            return sigma**2 * np.exp(-d / rho)
        if nu == 1.5:
            K = np.sqrt(np.array(3)) * d / rho
            return (1.0 + K) * np.exp(-K) * sigma**2
        if nu == 2.5:
            K = np.sqrt(np.array(5)) * d / rho
            return (1 + K + 1 / 3 * K**2) * np.exp(-K) * sigma**2
        term1 = 2 ** (1 - nu) / gamma(nu)
        term2 = (np.sqrt(np.array(2 * nu)) * d / rho) ** nu
        # changed back to kv (faster) but I assume you changed it for a reason. Can discuss next time.

        if BackendManager.backend_name == "torch":
            term3 = kv(nu, numpy.sqrt(2 * nu) * d.cpu() / rho)
            term3 = term3.to(device=BackendManager.device)
        else:
            term3 = kv(nu, np.sqrt(2 * nu) * d / rho)
        val = sigma**2 * term1 * term2 * term3
        val[d == 0] = sigma**2
        return val

    def f(
        self,
        freq_grid: Union[list, np.ndarray],
        infsum_grid: Union[list, np.ndarray],
        d: int = 2,
    ):
        # TODO: include infinite sum grid
        freq_grid = np.stack(freq_grid, axis=0)
        sigma, rho, nu = self.sigma.value, self.rho.value, self.nu.value
        pi = np.pi
        s = np.sqrt(np.sum(np.power(freq_grid, 2), axis=0)) / (2 * pi)
        if nu != np.inf:
            sdf = (
                sigma**2
                / (4 * pi**2)
                * 4
                * pi
                * gamma(nu + 1)
                * (2 * nu) ** nu
                / (gamma(nu) * rho ** (2 * nu))
                * (2 * nu / rho**2 + 4 * pi**2 * s**2) ** (-(nu + 1))
            )
        else:
            sdf = (
                1
                / (4 * pi**2)
                * sigma**2
                * 2
                * pi
                * rho**2
                * np.exp(-2 * pi**2 * rho**2 * s**2)
            )
        return sdf

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()


class MaternCovarianceModelAnisotropic(CovarianceModel):
    def __init__(self):
        sigma = Parameter("sigma", (0.01, 1000))
        rho1 = Parameter("rho1", (0.01, 1000))
        rho2 = Parameter("rho2", (0.01, 1000))
        nu = Parameter("nu", (0.01, 100))
        parameters = Parameters([sigma, nu, rho1, rho2])
        super(MaternCovarianceModelAnisotropic, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        sigma, rho1, rho2, nu = (
            self.sigma.value,
            self.rho1.value,
            self.rho2.value,
            self.nu.value,
        )
        lags[0, ...] /= rho1
        lags[1, ...] /= rho2
        d = np.sqrt(np.sum(lags**2, axis=0))
        if nu == 1.5:
            K = np.sqrt(np.array(3)) * d
            return (1.0 + K) * np.exp(-K) * sigma**2
        term1 = 2 ** (1 - nu) / gamma(nu)
        term2 = (np.sqrt(np.array(2 * nu)) * d) ** nu
        # changed back to kv (faster) but I assume you changed it for a reason. Can discuss next time.

        if BackendManager.backend_name == "torch":
            term3 = kv(nu, numpy.sqrt(2 * nu) * d.cpu())
            term3 = term3.to(device=BackendManager.device)
        else:
            term3 = kv(nu, np.sqrt(2 * nu) * d)
        val = sigma**2 * term1 * term2 * term3
        val[d == 0] = sigma**2
        return val

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()


class MaternCovarianceModelFrederik(CovarianceModel):
    def __init__(self):
        sigma = Parameter("sigma", (0.01, 1000))
        rho = Parameter("rho", (0.01, 1000))
        nu = Parameter("nu", (0.01, 100))
        parameters = Parameters([sigma, nu, rho])
        super(MaternCovarianceModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        lags = np.stack(lags, axis=0)
        d = np.sqrt(np.sum(lags**2, axis=0))
        sigma, rho, nu = self.sigma.value, self.rho.value, self.nu.value
        if nu == 1.5:
            K = np.sqrt(3) * d / rho
            return (1.0 + K) * np.exp(-K) * sigma**2
        term1 = 2 ** (1 - nu) / gamma(nu)
        term2 = (2 * np.sqrt(nu) * d / rho / np.pi) ** nu
        # changed back to kv (faster) but I assume you changed it for a reason. Can discuss next time.
        term3 = kv(nu, 2 * np.sqrt(nu) * d / rho / np.pi)
        val = sigma**2 * term1 * term2 * term3
        val[d == 0] = sigma**2
        return val

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()


def test_gradient_cov():
    """
    This test verifies that the analytical gradient of the covariance is close to a
    numerical approximation to that gradient.
    :return:
    """
    from numpy.testing import assert_allclose
    from .grids import RectangularGrid

    g = RectangularGrid((64, 64))
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    epsilon = 1e-3
    acv1 = model(g.lags_unique)
    model.rho = 10 + epsilon
    acv2 = model(g.lags_unique)
    g = model.gradient(
        g.lags_unique,
        Parameters(
            [
                model.rho,
            ]
        ),
    )["rho"]
    g2 = (acv2 - acv1) / epsilon
    assert_allclose(g, g2, rtol=1e-3)
