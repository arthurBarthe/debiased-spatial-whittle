from abc import ABC, abstractmethod, abstractproperty
import pickle
from debiased_spatial_whittle.backend import BackendManager

try:
    np = BackendManager.get_backend()
except:
    import numpy as np

import numpy
import param
from param import Parameterized
from param.parameterized import _get_param_repr

zeros = BackendManager.get_zeros()


def _parameterized_repr_html(p, open):
    """HTML representation for a Parameterized object"""
    if isinstance(p, Parameterized):
        cls = p.__class__
        title = cls.name + "()"
        value_field = "Value"
    else:
        cls = p
        title = cls.name
        value_field = "Default"

    tooltip_css = """
        .param-doc-tooltip{
          position: relative;
          cursor: help;
        }
        .param-doc-tooltip:hover:after{
          content: attr(data-tooltip);
          background-color: black;
          color: #fff;
          border-radius: 3px;
          padding: 10px;
          position: absolute;
          z-index: 1;
          top: -5px;
          left: 100%;
          margin-left: 10px;
          min-width: 250px;
        }
        .param-doc-tooltip:hover:before {
          content: "";
          position: absolute;
          top: 50%;
          left: 100%;
          margin-top: -5px;
          border-width: 5px;
          border-style: solid;
          border-color: transparent black transparent transparent;
        }
        """
    openstr = " open" if open else ""
    param_values = p.param.values().items()
    # contents = "".join(_get_param_repr(key, val, p.param[key])
    #                   for key, val in param_values)
    contents = ""
    for key, val in param_values:
        if key in ("name", "free_only"):
            continue
        if not p.param[key].readonly:
            contents += _get_param_repr(key, val, p.param[key])
        else:
            contents += (
                '<tr style="color:coral">'
                + _get_param_repr(key, val, p.param[key])[4:-7]
                + "<\tr>"
            )
    return (
        f"<style>{tooltip_css}</style>\n"
        f"<details {openstr}>\n"
        ' <summary style="display:list-item; outline:none;">\n'
        f"  <tt>{title}</tt>\n"
        " </summary>\n"
        ' <div style="padding-left:10px; padding-bottom:5px;">\n'
        '  <table style="max-width:100%; border:1px solid #AAAAAA;">\n'
        f'   <tr><th style="text-align:left;">Name</th><th style="text-align:left;">{value_field}</th><th style="text-align:left;">Type</th><th>Range</th></tr>\n'
        f"{contents}\n"
        "  </table>\n </div>\n</details>\n"
    )


class ModelParameter(param.Parameter):
    __slots__ = [
        "bounds",
    ]

    def __init__(self, *args, **kwargs):
        self.bounds = kwargs.pop("bounds")
        super().__init__(*args, allow_refs=True, per_instance=True, **kwargs)

    @property
    def free(self) -> bool:
        """true is the model parameter is free, i.e. not readonly and not constant"""
        return not (self.readonly or self.constant)


class ModelInterface(param.Parameterized):
    """
    Class defining the general interface for covariance models

    Attributes
    ----------

    """

    free_only = param.Boolean(per_instance=True, default=True)

    @abstractmethod
    def __call__(self, lags: np.ndarray):
        """
        Evaluate the covariance model at the passed lags.

        Parameters
        ----------
        lags
            array of lags. Shape (ndim, n1, ..., nk) where ndim is the number of spatial dimensions.

        Returns
        -------
        cov
            covariances. Shape (n1, ..., nk)
        """
        pass

    @property
    def free_parameters(self):
        """free parameters of the model - not deep"""
        out = []
        for p in self.param.objects().values():
            if isinstance(p, ModelParameter) and p.free:
                out.append(p.name)
        return out

    @property
    def n_free_parameters(self):
        """number of free parameters of the model - not deep"""
        return len(self.free_parameters)

    @abstractproperty
    def n_free_parameters_deep(self):
        """number of free parameters, recursive"""
        pass

    @abstractmethod
    def update_free_parameters(self, param_values: np.ndarray):
        """Update free parameters of the model recursively from array values.
        Useful for numerical optimization."""
        pass

    @abstractmethod
    def free_parameter_values_to_array_deep(self):
        """provide the free parameter values. Useful to pass to the x0 parameter of
        a numerical optimizer"""
        pass

    @abstractmethod
    def free_parameter_bounds_to_list_deep(self):
        """provide the free parameter bounds as a list. Useful to pass to bounds parameter of
        a numerical optimizer"""
        pass

    def set_param_bounds(self, bounds: dict[str, tuple[float, float]]):
        """set parameter bounds according to dictionary of parameter_name: parameter_bounds"""
        for k, v in bounds.items():
            self._set_param_bounds(k, v)

    def _set_param_bounds(self, param_name, bounds):
        """set parameter bounds for a single parameter. Checks that we make the bounds
        more restrictive"""
        left, right = getattr(self.param, param_name).bounds
        new_left, new_right = bounds
        if left is not None:
            if (new_left is None) or (new_left < left):
                raise ValueError("New bounds should not extend former bounds")
        if right is not None:
            if (new_right is None) or (new_right > right):
                raise ValueError("New bounds should not extend former bounds")
        setattr(getattr(self.param, param_name), "bounds", bounds)

    def link_param(self, param_name, other_param):
        """link a parameter to another. The former becomes readonly, and therefore is
        not free anymore"""
        setattr(self, param_name, other_param)
        setattr(getattr(self.param, param_name), "readonly", True)

    def fix_parameter(self, param_name):
        setattr(getattr(self.param, param_name), "constant", True)
        setattr(getattr(self.param, param_name), "readonly", True)

    def pickle(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def gradient(self, lags: np.ndarray, params: list[ModelParameter]) -> np.ndarray:
        """
        Compute the gradient of the model with respect to the passed parameters

        Parameters
        ----------
        lags


        params
            parameters for which we require the derivative

        Returns
        -------
        gradient
            last dimension indexes the parameters passed in params
        """
        grad = self._gradient(lags)
        out = []
        for p in params:
            out.append(grad[p.name])
        return np.stack(out, -1)

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def _repr_html_(self):
        pass

    def cov_mat_x1_x2(self, x1: np.ndarray, x2: np.ndarray = None) -> np.ndarray:
        """
        Compute the covariance matrix between between points in x1 and points in x2.

        Parameters
        ----------
        x1
            shape (n1, d), first set of locations
        x2
            shape (n2, d), second set of locations

        Returns
        -------
        covmat
            shape (n1, n2), covariance matrix
        """
        if x2 is None:
            x2 = x1
        x1 = np.expand_dims(x1, axis=1)
        x2 = np.expand_dims(x2, axis=0)
        lags = x1 - x2
        lags = np.transpose(lags, (2, 0, 1))
        return self(lags)


class CovarianceModel(ModelInterface):
    """
    Class to define low-level covariance modes (e.g. exponential, squared exponential).

    Attributes
    ----------

    """

    @property
    def n_free_parameters_deep(self):
        return len(self.free_parameters)

    def update_free_parameters(self, param_values: np.ndarray):
        """In the case of a simple model, we simply update the free parameters"""
        a, b = (
            param_values[: self.n_free_parameters],
            param_values[self.n_free_parameters :],
        )
        for p_name, value in zip(self.free_parameters, a):
            setattr(self, p_name, value)

    def free_parameter_values_to_array_deep(self):
        list_values = []
        for p in self.free_parameters:
            list_values.append(getattr(self, p))
        return np.array(list_values)

    def free_parameter_bounds_to_list_deep(self):
        list_bounds = []
        for p in self.free_parameters:
            list_bounds.append(getattr(self.param, p).bounds)
        return list_bounds

    def _repr_html_(self):
        return _parameterized_repr_html(self, True)

    def _compute(self, lags: np.ndarray):
        raise NotImplementedError()

    def __call__(self, lags: np.ndarray):
        out = self._compute(np.expand_dims(lags, -1))
        if out.shape[-1] == 1:
            out = np.squeeze(out, -1)
        return out

    def __add__(self, other):
        return SumModel(self, other)


class CompoundModel(ModelInterface):
    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children

    @property
    def n_free_parameters_deep(self):
        out = self.n_free_parameters
        for child in self.children:
            out += child.n_free_parameters_deep
        return out

    def update_free_parameters(self, param_values):
        a, b = (
            param_values[: self.n_free_parameters],
            param_values[self.n_free_parameters :],
        )
        for p_name, value in zip(self.free_parameters, a):
            setattr(self, p_name, value)
        # update parameters of children
        for child in self.children:
            child.update_free_parameters(b)
            b = b[child.n_free_parameters_deep :]

    def free_parameter_values_to_array_deep(self):
        list_values = []
        for p in self.free_parameters:
            list_values.append(getattr(self, p))
        array_values = np.array(list_values)
        return np.concatenate(
            [
                array_values,
            ]
            + [child.free_parameter_values_to_array_deep() for child in self.children]
        )

    def free_parameter_bounds_to_list_deep(self):
        list_bounds = []
        for p in self.free_parameters:
            list_bounds.append(getattr(self.param, p).bounds)
        for child in self.children:
            list_bounds.extend(child.free_parameter_bounds_to_list_deep())
        return list_bounds

    def _repr_html_(self):
        return (
            _parameterized_repr_html(self, True)
            + '<div style="margin-left:15px;padding-left:75px; border-left:solid gray 5px">'
            + "".join([child._repr_html_() for child in self.children])
            + "</div>"
        )

    def _compute(self, lags: np.ndarray):
        raise NotImplementedError()

    def __call__(self, lags: np.ndarray):
        out = self._compute(np.expand_dims(lags, -1))
        if out.shape[-1] == 1:
            out = np.squeeze(out, -1)
        return out

    def __add__(self, other):
        return SumModel(self, other)


class SumModel(CompoundModel):
    """
    Implements a covariance model defined as the sum of two several covariance models.

    Examples
    --------
    >>> model_1 = SquaredExponentialModel(rho=32)
    >>> model_2 = ExponentialModel(rho=5)
    >>> model = model_1 + model_2
    """

    def __new__(cls, *args, **kwargs):
        children = []
        for child in args:
            if isinstance(child, SumModel):
                children.append(child.children)
            else:
                children.append(child)
        return super(SumModel, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        children = args
        super().__init__(children, **kwargs)

    def _compute(self, lags: np.ndarray):
        values = (child._compute(lags) for child in self.children)
        out = sum(values)
        return out


class ExponentialModel(CovarianceModel):
    """
    Implements the Exponential covariance model.

    Attributes
    ----------
    rho: float
        length scale parameter

    sigma: float
        amplitude parameter

    Examples
    --------
    >>> model = ExponentialModel(rho=5, sigma=1.41)
    >>> model(np.array([[0., 1.], [0., 0.]]))
    array([1.9881    , 1.62771861])
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
    rho: float
        length scale parameter

    sigma: float
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
    rho: float
        length scale parameter of the kernel

    sigma: float
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
    rho: float
        length scale parameter of the kernel

    sigma: float
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
    rho: float
        length scale parameter of the kernel

    alpha: float
        alpha parameter of the kernel

    sigma: float
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
    base_model
        Covariance model

    eta
        Scaling factor

    phi
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
    >>> base_model = ExponentialModel(rho=12.)
    >>> bivariate_model = BivariateUniformCorrelation(base_model, r=0.3, f=2.)
    """

    r = ModelParameter(default=0.0, bounds=(-0.99, 0.99), doc="Correlation")
    f = ModelParameter(default=1.0, bounds=(0, numpy.inf), doc="Amplitude ratio")

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

    def _compute(self, lags: np.ndarray):
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
        out = np.zeros(acv11.shape + (2, 2))
        out = BackendManager.convert(out)
        out[..., 0, 0] = acv11
        out[..., 1, 1] = acv11 * self.f**2
        out[..., 0, 1] = acv11 * self.r * self.f
        out[..., 1, 0] = acv11 * self.r * self.f
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
        temp = np.stack((acv11 * self.f, acv11 * self.r), axis=-1)
        gradient_12 = np.concatenate(
            (temp, gradient_base_model * self.r * self.f), axis=-1
        )
        # gradient 21
        gradient_21 = gradient_12
        # gradient 22
        temp = np.stack((np.zeros_like(acv11), 2 * self.f * acv11), axis=-1)
        gradient_22 = np.concatenate((temp, gradient_base_model * self.f**2), axis=-1)
        row1 = np.stack((gradient_11, gradient_12), axis=x.ndim - 1)
        row2 = np.stack((gradient_21, gradient_22), axis=x.ndim - 1)
        return np.stack((row1, row2), axis=x.ndim - 1)


# TODO temporary fix
TMultivariateModel = None
SquaredModel = None
ChiSquaredModel = None
SeparableModel = None
