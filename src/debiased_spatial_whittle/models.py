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

    def __init_subclass__(cls, **kwargs):
        call_method = cls.__call__

        def new_call(self, lags: np.ndarray):
            out = call_method(self, np.expand_dims(lags, -1))
            out = np.squeeze(out)
            return out

        cls.__call__ = new_call

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
        acv = self._compute(lags)
        return acv


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
        acv = self._compute(lags)
        return acv


class SumModel(CompoundModel):
    """Class that allows to define a new model as the sum of several models."""

    sigma = ModelParameter(default=1.0, bounds=(0, numpy.infty))

    def __init__(self, children, *args, **kwargs):
        super().__init__(children, *args, **kwargs)

    def _compute(self, lags: np.ndarray):
        values = (child(lags) for child in self.children)
        out = sum(values)
        return out / self._norm_constant() * self.sigma**2

    def _norm_constant(self):
        try:
            sigmas = np.stack([child.sigma for child in self.children])
        except TypeError:
            sigmas = np.array([child.sigma for child in self.children])
        out = np.sum(sigmas**2, axis=0)
        return out


class ExponentialModel(CovarianceModel):
    rho = ModelParameter(default=1.0, bounds=(0, numpy.infty), doc="Range parameter")
    sigma = ModelParameter(
        default=1.0, bounds=(0, numpy.infty), doc="Amplitude parameter"
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
    rho = ModelParameter(default=1.0, bounds=(0, numpy.infty), doc="Range parameter")
    sigma = ModelParameter(
        default=1.0, bounds=(0, numpy.infty), doc="Amplitude parameter"
    )

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


class NuggetModel(CompoundModel):
    """
    Class to define a covariance modle based on a latent covariance model, and amplitude parameter and a nugget
    parameter.

    Properties
    ----------
    sigma: ModelParameter
        standard deviation

    nugget: ModelParameter
        Proportion of variance explained by the nugget
    """

    sigma = ModelParameter(default=1.0, bounds=(0, numpy.infty), doc="Amplitude")
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
        return (
            np.all(lags == 0, 0) * self.nugget
            + (1 - self.nugget) * self.children[0](lags)
        ) * self.sigma**2


class BivariateUniformCorrelation(CompoundModel):
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

    r = ModelParameter(default=0.0, bounds=(-1, 1), doc="Correlation")
    f = ModelParameter(default=1.0, bounds=(0, numpy.infty), doc="Amplitude ratio")

    def __init__(self, base_model: CovarianceModel):
        super(BivariateUniformCorrelation, self).__init__(
            [
                base_model,
            ]
        )

    @property
    def base_model(self):
        return self.children[0]

    @base_model.setter
    def base_model(self, model):
        raise AttributeError("Base model cannot be set")

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
            Covariance values with shape (m1, m2, ..., mk, 2, 2)

        """
        acv11 = self.base_model(lags)
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
