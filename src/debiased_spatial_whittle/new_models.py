from abc import ABC, abstractmethod, abstractproperty
import pickle
from debiased_spatial_whittle.backend import BackendManager
try:
    np = BackendManager.get_backend()
except:
    import numpy as np
import param
from param import Parameterized
from param.parameterized import _get_param_repr

zeros = BackendManager.get_zeros()


def _parameterized_repr_html(p, open):
    """HTML representation for a Parameterized object"""
    if isinstance(p, Parameterized):
        cls = p.__class__
        title = cls.name + "()"
        value_field = 'Value'
    else:
        cls = p
        title = cls.name
        value_field = 'Default'

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
        if key in ('name', 'free_only'):
            continue
        if not p.param[key].readonly:
            contents += _get_param_repr(key, val, p.param[key])
        else:
            contents += '<tr style="color:coral">' + _get_param_repr(key, val, p.param[key])[4:-7] + '<\tr>'
    return (
        f'<style>{tooltip_css}</style>\n'
        f'<details {openstr}>\n'
        ' <summary style="display:list-item; outline:none;">\n'
        f'  <tt>{title}</tt>\n'
        ' </summary>\n'
        ' <div style="padding-left:10px; padding-bottom:5px;">\n'
        '  <table style="max-width:100%; border:1px solid #AAAAAA;">\n'
        f'   <tr><th style="text-align:left;">Name</th><th style="text-align:left;">{value_field}</th><th style="text-align:left;">Type</th><th>Range</th></tr>\n'
        f'{contents}\n'
        '  </table>\n </div>\n</details>\n'
    )


class ModelParameter(param.Parameter):
    __slots__ = ['bounds', ]

    def __init__(self, *args, **kwargs):
        self.bounds = kwargs.pop('bounds')
        super().__init__(*args, allow_refs=True, per_instance=True, **kwargs)


class ModelInterface(param.Parameterized):
    free_only = param.Boolean(per_instance=True, default=True)

    @abstractmethod
    def __call__(self, lags: np.ndarray):
        pass

    @property
    def free_parameters(self):
        """free parameters of the model"""
        out = []
        for p in self.param.objects().values():
            if (not p.readonly) and (not p.constant):
                if isinstance(p, ModelParameter):
                    out.append(p.name)
        return out

    @property
    def n_free_parameters(self):
        """number of free parameters of the model"""
        return len(self.free_parameters)

    @abstractproperty
    def n_free_parameters_deep(self):
        """number of free parameters, recursive"""
        pass

    @abstractmethod
    def update_free_parameters(self, param_values: np.ndarray):
        """Update free parameters of the model recursively from array values.
        Useful for optimization."""
        pass

    @abstractmethod
    def free_parameter_values_to_array_deep(self):
        """provide the free parameter values. Useful to pass to the x0 of
        an optimizer"""
        pass

    @abstractmethod
    def free_parameter_bounds_to_list_deep(self):
        """provide the free parameter bounds as a list. Useful to pass to the x0 of
        an optimizer"""
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
                raise ValueError('New bounds should not extend former bounds')
        if right is not None:
            if (new_right is None) or (new_right > right):
                raise ValueError('New bounds should not extend former bounds')
        setattr(getattr(self.param, param_name), 'bounds', bounds)

    def link_param(self, param_name, other_param):
        """link a parameter to another. The former becomes readonly, and therefore is
        not free anymore"""
        setattr(self, param_name, other_param)
        setattr(getattr(self.param, param_name), 'readonly', True)

    def fix_parameter(self, param_name):
        setattr(getattr(self.param, param_name), 'constant', True)
        setattr(getattr(self.param, param_name), 'readonly', True)

    def pickle(self, file: str):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def _repr_html_(self):
        pass


class Model(ModelInterface):
    @property
    def n_free_parameters_deep(self):
        return len(self.free_parameters)

    def update_free_parameters(self, param_values: np.ndarray):
        """In the case of a simple model, we simply update the free parameters"""
        a, b = param_values[:self.n_free_parameters], param_values[self.n_free_parameters:]
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
        lags = np.expand_dims(lags, -1)
        acv = self._compute(lags)
        if acv.shape[-1] == 1:
            return np.squeeze(acv, -1)
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
        a, b = param_values[:self.n_free_parameters], param_values[self.n_free_parameters:]
        for p_name, value in zip(self.free_parameters, a):
            setattr(self, p_name, value)
        # update parameters of children
        for child in self.children:
            child.update_free_parameters(b)
            b = b[child.n_free_parameters_deep:]

    def free_parameter_values_to_array_deep(self):
        list_values = []
        for p in self.free_parameters:
            list_values.append(getattr(self, p))
        array_values = np.array(list_values)
        return np.concatenate(
            [array_values, ] + [child.free_parameter_values_to_array_deep() for child in self.children])

    def free_parameter_bounds_to_list_deep(self):
        list_bounds = []
        for p in self.free_parameters:
            list_bounds.append(getattr(self.param, p).bounds)
        for child in self.children:
            list_bounds.extend(child.free_parameter_bounds_to_list_deep())
        return list_bounds

    def _repr_html_(self):
        return (_parameterized_repr_html(self, True) +
                '<div style="margin-left:15px;padding-left:75px; border-left:solid gray 5px">' +
                ''.join([child._repr_html_() for child in self.children]) +
                '</div>')


class SumModel(CompoundModel):
    """Class that allows to define a new model as the sum of several models."""
    sigma = ModelParameter(default=1.0, bounds=(0, None))

    def __init__(self, children, *args, **kwargs):
        super().__init__(children, *args, **kwargs)

    def __call__(self, lags: np.ndarray):
        values = (child(lags) for child in self.children)
        out = sum(values)
        return out / self._norm_constant() * self.sigma ** 2

    def _norm_constant(self):
        try:
            sigmas = np.stack([child.sigma for child in self.children])
        except TypeError:
            sigmas = np.array([child.sigma for child in self.children])
        out = np.sum(sigmas ** 2, axis=0)
        return out


class ExponentialModel(Model):
    rho = ModelParameter(default=1., bounds=(0, None), doc='Range parameter')
    sigma = ModelParameter(default=1., bounds=(0, 1), doc='Amplitude parameter')

    def _compute(self, lags: np.ndarray):
        d = np.sqrt(np.sum(lags ** 2, 0)) / self.rho
        return self.sigma ** 2 * np.exp(- d)


class SquaredExponentialModel(Model):
    rho = ModelParameter(default=1., bounds=(0, None), doc='Range parameter')
    sigma = ModelParameter(default=1., bounds=(0, 1), doc='Amplitude parameter')

    def _compute(self, lags: np.ndarray):
        d = np.sum(lags ** 2, 0) / (2 * self.rho ** 2)
        return self.sigma ** 2 * np.exp(- d)


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
    sigma = ModelParameter(default=1., bounds=(0, None), doc='Amplitude')
    nugget = ModelParameter(default=0., bounds=(0, 1), doc='Nugget amplitude')

    def __init__(self, model, *args, **kwargs):
        super().__init__([model, ], *args, **kwargs)

    def _compute(self, lags: np.ndarray):
        return (np.all(lags == 0, 0) * self.nugget + (1 - self.nugget) * self.children[0](lags)) * self.sigma ** 2