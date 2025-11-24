from abc import ABC, abstractmethod, abstractproperty
import pickle
from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()

import numpy
import param
from param import Parameterized
from param.parameterized import _get_param_repr

zeros = BackendManager.get_zeros()
inv = BackendManager.get_inv()


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
    """
    Class used to represent a covariance model's parameter. When writing a new model, any parameter should be of
    this type. See for instance the models defined in the univariate module.

    Attributes
    ----------
    bounds: tuple[float, float]
        Range of possible parameter values. Used by the optimizers during inference.

    default: float
        Default value of the parameter.

    free: bool
        True if the parameter is free, i.e. can be modified. By default all parameters are free. However one can
        fix a model parameter using the fix_parameter method of ModelInterface. In particular, this is relevant
        for optimization: the optimizers will only work with the free parameters.
    """
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
    Class defining the general interface for covariance models.

    Attributes
    ----------

    """
    @abstractmethod
    def __call__(self, lags: xp.ndarray):
        """
        Evaluate the covariance model at the passed lags.

        Parameters
        ----------
        lags
            array of lags. Shape (ndim, n1, ..., nk) where ndim is the number of spatial dimensions.

        Returns
        -------
        cov
            covariances at the passed lags. The shape of cov will depend on scalar vs multivariate model whether
            vectorized models are used. This is summarized by the table below, where $p$ denotes the
            number of variates of the random field ($p=1$ for a scalar random field), and $m$ denotes
            the number of models in the case of vectorized models.

            cov shape | p=1 | p > 1
            :----------- |:-------------:| -----------:
            single model         | (n1, ..., nk)        | (n1, ..., nk, p, p)
            vectorized model         | (n1, ... nk, m)        | (n1, ..., nk, m, p, p)
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
    def update_free_parameters(self, param_values: xp.ndarray):
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

    def gradient(self, lags: xp.ndarray, params: list[ModelParameter]) -> xp.ndarray:
        """
        Compute the gradient of the model with respect to the passed parameters

        Parameters
        ----------
        lags
            array of lags. Shape (d, n1, ..., nk)

        params
            parameters for which we require the derivative

        Returns
        -------
        gradient
            array of gradient w.r.t. parameters in params. The shape depends on scalar versus multivariate
            random field model and on unique versus vectorized model. This is summarized by the table below,
            where $p$ denotes the number of variates of the random field ($p=1$ for a scalar random field),
            and $m$ denotes the number of models in the case of vectorized models. $g$ denotes the number of parameters
            in params, those for which we request the gradient.

            cov shape | p=1 | p > 1
            :----------- |:-------------:| -----------:
            single model         | (n1, ..., nk, g)        | (n1, ..., nk, g, p, p)
            vectorized model         | Not tested      | Not tested


        """
        grad = self._gradient(lags)
        out = []
        for p in params:
            out.append(grad[p.name])
        n_spatial_dims = lags.shape[0]
        return xp.stack(out, n_spatial_dims)

    def _gradient(self, lags: xp.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def _repr_html_(self):
        pass

    def cov_mat_x1_x2(self, x1: xp.ndarray, x2: xp.ndarray = None) -> xp.ndarray:
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
        x1 = xp.expand_dims(x1, axis=1)
        x2 = xp.expand_dims(x2, axis=0)
        lags = x1 - x2
        lags = xp.transpose(lags, (2, 0, 1))
        return self(lags)

    def predict(
        self,
        x_obs: xp.ndarray,
        y_obs: xp.ndarray,
        x_pred: xp.ndarray,
        return_variance: bool = False,
    ):
        """
        Compute conditional mean at a set of locations x_pred given values y_obs observed at x_obs.

        Parameters
        ----------
        x_obs
            shape (n_obs, d), array of locations where observations are made
        y_obs
            shape (n_obs, 1), observed values
        x_pred
            shape (n_pred, d), array of locations where predicted values are requested

        Returns
        -------
        y_pred
            shape (n_pred, 1), array of predicted values
        """
        x_obs = xp.expand_dims(x_obs, 1)
        # x_obs (n_obs, 1, d)
        lags_xx = x_obs - xp.transpose(x_obs, (1, 0, 2))
        # lags_xx (n_obs, n_obs, d)

        cov_mat_xx = self(xp.transpose(lags_xx, (2, 0, 1)))
        # cov_mat_xx (n_obs, n_obs)
        cov_mat_xx_inv = inv(cov_mat_xx)

        x_pred = xp.expand_dims(x_pred, 1)
        # x_pred (n_pred, 1, d)

        lags_yx = x_pred - xp.transpose(x_obs, (1, 0, 2))
        # lags_yx (n_pred, n_obs, d)

        sigma_yx = self(xp.transpose(lags_yx, (2, 0, 1)))
        # sigma_yx (n_pred, n_obs)

        weights = xp.dot(sigma_yx, cov_mat_xx_inv)
        # weights (n_pred, n_obs)
        y_pred = xp.matmul(weights, y_obs)
        return y_pred


class CovarianceModel(ModelInterface):
    """
    Class to define low-level covariance modes (e.g. exponential, squared exponential).
    """

    @property
    def n_free_parameters_deep(self):
        return len(self.free_parameters)

    def update_free_parameters(self, param_values: xp.ndarray):
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
        return xp.array(list_values)

    def free_parameter_bounds_to_list_deep(self):
        list_bounds = []
        for p in self.free_parameters:
            list_bounds.append(getattr(self.param, p).bounds)
        return list_bounds

    def _repr_html_(self):
        return _parameterized_repr_html(self, True)

    def _compute(self, lags: xp.ndarray):
        raise NotImplementedError()

    def __call__(self, lags: xp.ndarray):
        ndim = lags.ndim
        out = self._compute(xp.expand_dims(lags, -1))
        if out.shape[ndim - 1] == 1:
            out = xp.squeeze(out, ndim - 1)
        return out

    def __add__(self, other):
        return SumModel(self, other)


class CompoundModel(ModelInterface):
    """
    Class that permits to combine several covariance models to define a new covariance model. For instance, this
    allows to define a sum covariance model, or a space-time covariance model that uses a spatial component and
    a temporal component.
    """
    def __init__(self, children: list[ModelInterface], *args, **kwargs):
        """
        Parameters
        ----------
        children
            list of children covariance models
        """
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
        array_values = xp.array(list_values)
        return xp.concatenate(
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

    def _compute(self, lags: xp.ndarray):
        raise NotImplementedError()

    def __call__(self, lags: xp.ndarray):
        ndim = lags.ndim
        out = self._compute(xp.expand_dims(lags, -1))
        if out.shape[ndim - 1] == 1:
            out = xp.squeeze(out, ndim - 1)
        return out

    def __add__(self, other):
        return SumModel(self, other)


class SumModel(CompoundModel):
    """
    Implements a covariance model defined as the sum of two covariance models.

    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import SquaredExponentialModel, ExponentialModel
    >>> model_1 = SquaredExponentialModel(rho=32)
    >>> model_2 = ExponentialModel(rho=5)
    >>> model = model_1 + model_2
    >>> model.free_parameter_values_to_array_deep()
    array([32.,  1.,  5.,  1.])
    >>> model_1.rho = 30
    >>> model.free_parameter_values_to_array_deep()
    array([30.,  1.,  5.,  1.])
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

    def _compute(self, lags: xp.ndarray):
        values = (child._compute(lags) for child in self.children)
        out = sum(values)
        return out


# TODO temporary fix
TMultivariateModel = None
SquaredModel = None
ChiSquaredModel = None
SeparableModel = None