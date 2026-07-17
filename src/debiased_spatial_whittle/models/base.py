import logging
import copy
from abc import ABC, abstractmethod

import numpy as np

from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.caching import Freezable, ban_if_frozen

xp = BackendManager.get_backend()
inv = BackendManager.get_inv()

from torch.autograd.functional import jacobian
from copy import deepcopy

try:
    from rich import print
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ModelParameter:
    def __init__(self, default, bounds=(None, None), doc="", latex_display: str = None):
        self.default = BackendManager.to_device(xp.squeeze(xp.asarray(default)).astype(xp.float64))
        self.bounds = bounds
        self.doc = doc
        self.latex_display = latex_display

    def __set_name__(self, owner, name):
        self.name = name
        if not '_parameters' in owner.__dict__:
            owner._parameters = []
        owner._parameters.append(name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(f"_{self.name}", self.default)

    def __set__(self, obj, value):
        if hasattr(obj, '_frozen_parameters') and self.name in obj._frozen_parameters:
            raise ValueError(f"Parameter {self.name} is frozen and cannot be set.")
        if hasattr(obj, 'frozen') and obj.frozen:
            raise ValueError(f"Parameter {self.name} cannot be set at the model is frozen.")
        if value is not None:
            obj.__dict__[f"_{self.name}"] = BackendManager.to_device(xp.squeeze(xp.asarray(value)).astype(xp.float64))



class ModelInterface(ABC):
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @name.setter
    @abstractmethod
    def name(self, value):
        raise NotImplementedError()

    @abstractmethod
    def copy(self):
        raise NotImplementedError()

    @abstractmethod
    def frozen_copy(self):
        raise NotImplementedError()

    @property
    def parameters(self) -> tuple:
        return self.get_parameters(self.parameter_names)

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def free_parameters(self) -> tuple:
        return self.get_parameters(self.free_parameter_names)

    @property
    def n_free_parameters(self):
        return len(self.free_parameters)

    @property
    @abstractmethod
    def free_parameter_bounds(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def parameter_names(self) -> tuple:
        raise NotImplementedError()

    @property
    @abstractmethod
    def free_parameter_names(self) -> tuple:
        raise NotImplementedError()

    @property
    @abstractmethod
    def parameters_repr(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def free_parameters_repr(self):
        raise NotImplementedError()

    @abstractmethod
    def get_parameter(self, name: str):
        raise NotImplementedError()

    @abstractmethod
    def set_parameter(self, name, value) -> bool:
        raise NotImplementedError()

    def get_parameters(self, names: list[str]) -> tuple:
        return tuple([self.get_parameter(name) for name in names])

    def set_parameters(self, name_values: dict[str, object]):
        for param_name, param_value in name_values.items():
            self.set_parameter(param_name, param_value)

    @abstractmethod
    def set_parameter_bounds(self, name: str, bounds: tuple[float, float]) -> None:
        raise NotImplementedError()

    # methods useful for optimizers --------------------

    def update_free_parameters(self, values):
        free_parameter_names = self.free_parameter_names
        self.set_parameters(dict(zip(free_parameter_names, values)))

    def free_parameter_values_to_array_deep(self):
        return xp.array(self.free_parameters)

    def free_parameter_bounds_to_list_deep(self):
        return self.free_parameter_bounds

    # --------------------------------------------------

    @abstractmethod
    def freeze_parameter(self, name, value) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Here we expect the model parameters to be passed via param_args.
        param_args: can be a tuple (should have size n_parameters) or a named tuple (not implemented yet)
        """
        raise NotImplementedError()

    def __call__(self, lags: xp.ndarray) -> xp.ndarray:
        ndim = lags.ndim
        lags = xp.expand_dims(lags, -1)
        params = self.parameters
        out = self.compute(lags, *params)
        if out.shape[ndim - 1] == 1:
            out = xp.squeeze(out, ndim - 1)
        return out

    def cov_mat_x1_x2(self, x1: xp.ndarray, x2: xp.ndarray = None):
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

    def jacobian(self, lags: xp.ndarray, param_names: tuple[str] = None) -> xp.ndarray:
        if BackendManager.backend_name != "torch":
            return self.jacobian_scipy(lags, param_names)
        if param_names is None:
            param_names = self.parameter_names
        param_values = self.get_parameters(param_names)

        def func(*args):
            full_args = []
            for param_name in self.parameter_names:
                if param_name in param_names:
                    full_args.append(args[param_names.index(param_name)])
                else:
                    full_args.append(self.parameters[self.parameter_names.index(param_name)])
            return self.compute(lags, *full_args)

        out = jacobian(func, param_values, strategy="forward-mode", vectorize=True)
        return dict(zip(param_names, out))

    def jacobian_scipy(self, lags: xp.ndarray, param_names: tuple[str] = None) -> xp.ndarray:
        """
        Examples
        --------
        >>> from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
        >>> model = SquaredExponentialModel(rho=32, name="model")
        >>> import numpy as np
        >>> lags = np.array([[0., 0.], [0., 1.]])
        >>> model.jacobian_scipy(lags)
        """
        from scipy.differentiate import jacobian
        if param_names is None:
            param_names = self.parameter_names
        param_values = self.get_parameters(param_names)

        def func(x):
            full_args = []
            for param_name in self.parameter_names:
                if param_name in param_names:
                    full_args.append(x[param_names.index(param_name)])
                else:
                    full_args.append(self.parameters[self.parameter_names.index(param_name)])
            return self.compute(np.expand_dims(lags, -1), *full_args)

        def func2(x):
            shape = x.shape[1:]
            return np.apply_along_axis(func, axis=0, arr=x).reshape((-1, ) + shape)

        out = jacobian(func2, param_values)
        acv_shape = self.compute(lags, *self.parameters).shape
        return dict(zip(param_names, out.df.T.reshape((len(param_names),) + acv_shape)))

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other):
        raise NotImplementedError()




class CovarianceModel(ModelInterface, Freezable):

    def __init__(self, children: tuple[ModelInterface], *params, name: str = None):
        self.name = name
        self._frozen_parameters = []
        self.children = children
        self.assign_params(*params)
        super().__init__()
        self._parameter_bounds = self._init_parameter_bounds()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_parameters"):
            cls._parameters = []

    def _init_parameter_bounds(self):
        bounds = dict()
        for pname in self._parameters:
            bounds[pname] = getattr(self.__class__, pname).bounds
        return bounds

    def assign_params(self, *params):
        for param_name, param_value in zip(self._parameters, params):
            setattr(self, param_name, param_value)

    @property
    def name(self):
        return self._name

    @name.setter
    @ban_if_frozen
    def name(self, value):
        self._name = value if value else self.__class__.__name__

    def copy(self):
        return Freezable.copy(self)
    
    def frozen_copy(self):
        return Freezable.frozen_copy(self)

    @property
    def display_subscript(self) -> str:
        if not hasattr(self, "_display_subscript"):
            return None
        return self._display_subscript

    @display_subscript.setter
    def display_subscript(self, value: str):
        self._display_subscript = value

    @property
    def free_parameter_bounds(self):
        out = []
        for param_name in self._parameters:
            if not param_name in self._frozen_parameters:
                out.append(self._parameter_bounds[param_name])
        for child in self.children:
            out.extend(child.free_parameter_bounds)
        return out

    @ban_if_frozen
    def set_parameter_bounds(self, name: str, bounds: tuple[float, float]) -> None:
        """
        Set the parameter bounds.
        """
        if name.count("_") == 0:
            if name in self._parameters:
                self._parameter_bounds[name] = bounds
                return True
            return False
        model_name, param_name = name.split("_", 1)
        if model_name == self.name:
            self._parameter_bounds[param_name] = bounds
            return True
        else:
            for child in self.children:
                value = child.set_parameter_bounds(name, bounds)
                if value:
                    return True
            return False

    @property
    def parameter_names(self) -> tuple[str]:
        out = []
        for param_name in self._parameters:
            out.append(f'{self.name}_{param_name}')
        for child in self.children:
            out.extend(child.parameter_names)
        return out

    @property
    def free_parameter_names(self) -> tuple[str]:
        out = []
        for param_name in self._parameters:
            if not param_name in self._frozen_parameters:
                out.append(f'{self.name}_{param_name}')
        for child in self.children:
            out.extend(child.free_parameter_names)
        return tuple(out)

    @property
    def parameters_repr(self) -> tuple[str]:
        out = []
        for pname in self._parameters:
            if self.display_subscript is not None:
                out.append(rf"{getattr(self.__class__, pname).latex_display}_{self.display_subscript}")
            else:
                out.append(getattr(self.__class__, pname).latex_display)
        for child in self.children:
            out.extend(child.parameters_repr)
        return out

    @property
    def free_parameters_repr(self) -> tuple[str]:
        out = []
        parameters_rep = self.parameters_repr
        parameter_names = self.parameter_names
        free_parameter_names = self.free_parameter_names
        for pname, prepr in zip(parameter_names, parameters_rep):
            if pname in free_parameter_names:
                out.append(prepr)
        return out

    def get_parameter(self, name: str):
        model_name, param_name = name.split("_", 1)
        if model_name == self.name:
            return getattr(self, param_name)
        else:
            for child in self.children:
                value = child.get_parameter(name)
                if value is not None:
                    return value
            return None

    @ban_if_frozen
    def set_parameter(self, name, value) -> bool:
        model_name, param_name = name.split("_", 1)
        if model_name == self.name:
            setattr(self, param_name, value)
            logging.debug(f"Set {param_name} to: {value} in {model_name}")
            return True
        else:
            for child in self.children:
                assigned = child.set_parameter(name, value)
                if assigned:
                    return True
            return False

    @ban_if_frozen
    def freeze_parameter(self, name):
        if name.count("_") == 0:
            if name in self._parameters:
                self._frozen_parameters.append(name)
                return True
            return False
        model_name, param_name = name.split("_", 1)
        if model_name == self.name:
            self._frozen_parameters.append(param_name)
            return True
        else:
            for child in self.children:
                value = child.freeze_parameter(name)
                if value:
                    return True
            return False

    def __add__(self, other):
        return SumModel(self, other)

    def __mul__(self, other):
        return ProductModel(self, other)

    def _split_children_params(self, *params):
        out = []
        for child in self.children:
            n_params = child.n_parameters
            temp, params = params[:n_params], params[n_params:]
            out.append(temp)
        return out

    def __repr__(self):
        """Text representation of the model showing tree structure, parameter names, values, and fixed status."""
        return self._repr_tree()
    
    def __rich__(self):
        """Rich terminal representation for use with rich library."""
        if RICH_AVAILABLE:
            from rich.tree import Tree
            class_name = self.__class__.__name__
            tree = Tree(f"[bold blue]{self.name}[/bold blue] ([dim]{class_name}[/dim])")
            self._build_rich_tree(tree)
            return tree
        return self._repr_tree()
    
    def _build_rich_tree(self, parent_tree):
        """Recursively build rich tree structure."""
        if not RICH_AVAILABLE:
            return
        
        # Add parameters
        if self._parameters:
            params_tree = parent_tree.add("[bold]Parameters[/bold]")
            for param_name in self._parameters:
                param_value = getattr(self, param_name)
                is_fixed = param_name in self._frozen_parameters
                fixed_text = " [red](frozen)[/red]" if is_fixed else ""
                # Add bounds if not frozen
                bounds_text = ""
                if not is_fixed and param_name in self._parameter_bounds:
                    bounds = self._parameter_bounds[param_name]
                    bounds_text = f" [dim](bounds: [{bounds[0]}, {bounds[1]}])[/dim]"
                params_tree.add(f"{param_name}: {param_value}{fixed_text}{bounds_text}")
        
        # Add children recursively
        for child in self.children:
            child_class_name = child.__class__.__name__
            child_tree = parent_tree.add(f"[bold cyan]{child.name}[/bold cyan] ([dim]{child_class_name}[/dim])")
            child._build_rich_tree(child_tree)
    
    def _repr_tree(self, prefix="", is_last=True):
        """Recursive helper for text representation."""
        lines = []
        connector = "└── " if is_last else "├── "
        class_name = self.__class__.__name__
        lines.append(f"{prefix}{connector}{self.name} ({class_name})")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # Parameters
        if self._parameters:
            for idx, param_name in enumerate(self._parameters):
                param_value = getattr(self, param_name)
                is_fixed = param_name in self._frozen_parameters
                fixed_marker = " (frozen)" if is_fixed else ""
                # Add bounds if not frozen
                bounds_info = ""
                if not is_fixed and param_name in self._parameter_bounds:
                    bounds = self._parameter_bounds[param_name]
                    bounds_info = f" [{bounds[0]}, {bounds[1]}]"
                is_last_param = (idx == len(self._parameters) - 1) and (not self.children)
                param_connector = "└── " if is_last_param else "├── "
                lines.append(f"{new_prefix}{param_connector}{param_name}: {param_value}{fixed_marker}{bounds_info}")
        
        # Children
        for idx, child in enumerate(self.children):
            is_last_child = (idx == len(self.children) - 1)
            lines.append(child._repr_tree(new_prefix, is_last_child))
        
        return '\n'.join(lines)

    def _repr_html_(self):
        """HTML representation of the model showing tree structure, parameter names, values, and fixed status."""
        html = []
        html.append('<div style="margin-left:15px;padding-left:10px;border-left:solid gray 2px;">')
        class_name = self.__class__.__name__
        html.append(f'<b>{self.name} ({class_name})</b>')
        
        # Parameters table
        if self._parameters:
            html.append('<table style="margin-left:15px;">')
            html.append('<tr><th style="text-align:left;">Parameter</th><th style="text-align:left;">Value</th><th>Fixed</th><th style="text-align:left;">Bounds</th></tr>')
            for param_name in self._parameters:
                param_obj = getattr(self.__class__, param_name)
                param_value = getattr(self, param_name)
                is_fixed = param_name in self._frozen_parameters
                fixed_str = 'Yes' if is_fixed else 'No'
                fixed_style = 'color:orange;' if is_fixed else ''
                # Add bounds column
                bounds_str = '-' if is_fixed else f'[{self._parameter_bounds[param_name][0]}, {self._parameter_bounds[param_name][1]}]'
                html.append(f'<tr><td>{param_name}</td><td>{param_value}</td><td style="{fixed_style}">{fixed_str}</td><td>{bounds_str}</td></tr>')
            html.append('</table>')
        
        # Children
        for child in self.children:
            html.append(child._repr_html_())
        
        html.append('</div>')
        return '\n'.join(html)

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

    # ------------ backward compatibility ---------
    def fix_parameter(self, param_name: str):
        self.freeze_parameter(f'{self.name}_{param_name}')


class BaseCovarianceModel(CovarianceModel):
    def __init__(self, *params, name=None):
        super().__init__((), *params, name=name)


class SumModel(CovarianceModel):
    """
    A covariance model that represents the sum of multiple covariance models.
    The compute method returns the sum of the covariances of the children.
    """
    _parameters = []
    
    def __init__(self, *models, name: str = None):
        # SumModel itself has no parameters, only children
        children = []
        for child in models:
            if isinstance(child, SumModel):
                children.extend(child.children)
            else:
                children.append(child)
        super().__init__(children, name=name)

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Compute the sum of covariances from all child models.
        """
        child_params = self._split_children_params(*params)
        result = self.children[0].compute(lags, *child_params[0])
        for child, child_param in zip(self.children[1:], child_params[1:]):
            result += child.compute(lags, *child_param)
        return result


class Sum2Models(CovarianceModel):
    theta = ModelParameter(default=xp.pi / 4, bounds=(0, xp.pi / 2), latex_display=r"\theta")

    def __init__(self, model1: CovarianceModel, model2: CovarianceModel, theta: float = None, name: str = None):
        super().__init__((model1, model2), theta, name=name)

    def compute(self, lags: xp.ndarray, theta, *params) -> xp.ndarray:
        children_params = self._split_children_params(*params)
        acv1 = self.children[0].compute(lags, *children_params[0])
        acv2 = self.children[1].compute(lags, *children_params[1])
        return xp.cos(theta) * acv1 + xp.sin(theta) * acv2


class ProductModel(CovarianceModel):
    """
    A covariance model that represents the product of multiple covariance models.
    The compute method returns the product of the covariances of the children.
    """
    _parameters = []
    
    def __init__(self, *models, name: str = None):
        # ProductModel itself has no parameters, only children
        children = []
        for child in models:
            if isinstance(child, ProductModel):
                children.extend(child.children)
            else:
                children.append(child)
        super().__init__(children, name=name)

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Compute the product of covariances from all child models.
        """
        child_params = self._split_children_params(*params)
        result = self.children[0].compute(lags, *child_params[0])
        for child, child_param in zip(self.children[1:], child_params[1:]):
            result *= child.compute(lags, *child_param)
        return result


class ReparameterizedModel(ModelInterface, ABC, Freezable):
    """
    Class that allows to use an alternative parameterization of a base model.
    """
    def __init__(self, base_model: ModelInterface, name: str = None):
        self.base_model = base_model
        self._parameter_bounds = []
        self._frozen_parameters = []
        self.name = name
        super().__init__()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        name = name if name else self.__class__.__name__
        self._name = name

    def copy(self):
        duplicate = Freezable.copy(self)
        duplicate.base_model.unfreeze()
        return duplicate

    def frozen_copy(self):
        return Freezable.frozen_copy(self)

    @abstractmethod
    def map_parameters(self, *params):
        raise NotImplementedError()

    @abstractmethod
    def imap_parameters(self, *base_model_params):
        raise NotImplementedError()

    @property
    def parameters(self):
        return self.imap_parameters(*self.base_model.parameters)

    @property
    def n_parameters(self):
        return len(self.parameters)

    def get_parameter(self, name: str) -> xp.ndarray:
        return self.parameters[self.base_model.parameter_names.index(name)]

    def set_parameter(self, name: str, value: xp.ndarray):
        current_parameters = self.parameters
        new_parameters = [current_parameters[i] if name != pname else value for i, pname in enumerate(self.parameter_names)]
        mapped_parameters = self.map_parameters(*new_parameters)
        mapped_parameter = mapped_parameters[self.parameter_names.index(name)]
        self.base_model.set_parameter(name, mapped_parameter)

    def set_parameter_bounds(self, name: str, bounds: tuple[float, float]) -> None:
        self._parameter_bounds[name] = bounds

    @property
    def free_parameter_bounds(self):
        return [self._parameter_bounds[pname] for pname in self.free_parameter_names]

    @property
    def n_free_parameters(self):
        return len(self.free_parameters)

    @property
    def parameter_names(self):
        raise NotImplementedError()

    @property
    def free_parameter_names(self) -> tuple:
        return tuple(filter(lambda pname: pname not in self._frozen_parameters, self.parameter_names))

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        base_model_params = self.map_parameters(*params)
        return self.base_model.compute(lags, *base_model_params)

    def __add__(self, other):
        return SumModel(self, other)

    def __mul__(self, other):
        return ProductModel(self, other)


class LogScaleReparameterizedModel(ReparameterizedModel):
    """
    Class that allows to use a log scale parameterization of a base model. One can specify which
    parameters use the log scale representation via the sel argument.
    """
    def __init__(self, base_model: ModelInterface, sel: tuple[bool] = None):
        super().__init__(base_model)
        self.sel = xp.array(sel).astype(bool) if sel else xp.ones(self.base_model.n_parameters).astype(bool)

    def map_parameters(self, *params):
        mapped_params = []
        for log, param in zip(self.sel, params):
            param = xp.asarray(param)
            if log:
                mapped_params.append(xp.exp(param))
            else:
                mapped_params.append(param)
        return tuple(mapped_params)

    def imap_parameters(self, *params):
        mapped_params = []
        for log, param in zip(self.sel, params):
            param = xp.asarray(param)
            if log:
                mapped_params.append(xp.log(param))
            else:
                mapped_params.append(param)
        return tuple(mapped_params)

    @property
    def parameter_names(self):
        return self.base_model.parameter_names

    @property
    def free_parameter_names(self):
        return self.base_model.free_parameter_names

    def freeze_parameter(self, name):
        self.base_model.freeze_parameter(name)

    @property
    def free_parameters_repr(self):
        return tuple([f"log {p_repr}" if self.sel[self.parameter_names.index(p_name)] else p_repr
                      for (p_repr, p_name) in zip(self.base_model.free_parameters_repr, self.free_parameter_names)])

    @property
    def parameters_repr(self):
        repr_base_params = self.base_model.parameters_repr
        return tuple([f"log {p_repr}" if self.sel[i] else p_repr
                      for (i, p_repr) in enumerate(repr_base_params)])

    def frozen_copy(self):
        copy = self.copy()
        copy.freeze()
        return copy

    @property
    def free_parameter_bounds(self):
        base_bounds = self.base_model.free_parameter_bounds
        free_sel = [sel_i for (pname, sel_i) in zip(self.parameter_names, self.sel) if pname in self.free_parameter_names]
        mapped_bounds = []
        for i, (lower, upper) in enumerate(base_bounds):
            if free_sel[i]:
                # Log scale: transform bounds using log
                # Convert to backend type, apply log, then convert back to Python float
                lower_t = xp.log(xp.asarray(lower))
                upper_t = xp.log(xp.asarray(upper))
                mapped_bounds.append((lower_t.item() if hasattr(lower_t, 'item') else float(lower_t),
                                      upper_t.item() if hasattr(upper_t, 'item') else float(upper_t)))
            else:
                # No transformation
                mapped_bounds.append((lower, upper))
        return mapped_bounds


class SigmoidReparameterizedModel(ReparameterizedModel, Freezable):
    """
    Class that applies a sigmoid transformation to map unbounded parameters to bounded ones.
    
    The transformation maps from (-inf, inf) to the bounds of each parameter in the base model.
    For a parameter with bounds (a, b), the mapping is:
        x -> a + (b - a) * sigmoid(x)
    where sigmoid(x) = 1 / (1 + exp(-x))
    
    The inverse mapping is:
        y -> logit((y - a) / (b - a))
    where logit(p) = log(p / (1 - p))
    """
    def __init__(self, base_model: ModelInterface, sel: tuple[bool] = None):
        super().__init__(base_model)
        self.sel = xp.array(sel).astype(xp.bool) if sel else xp.ones(self.base_model.n_parameters).astype(bool)
        # Store the bounds for mapping
        self._param_bounds = self.base_model.free_parameter_bounds
    
    def map_parameters(self, *params):
        """Map from unbounded space to bounded space using sigmoid."""
        mapped_params = []
        for i, (param, use_sigmoid) in enumerate(zip(params, self.sel)):
            if use_sigmoid:
                lower, upper = self._param_bounds[i]
                # Convert to backend array and apply sigmoid
                param_t = xp.asarray(param)
                sigmoid_val = 1.0 / (1.0 + xp.exp(-param_t))
                mapped_param = lower + (upper - lower) * sigmoid_val
                mapped_params.append(mapped_param)
            else:
                mapped_params.append(param)
        return tuple(mapped_params)
    
    def imap_parameters(self, *params):
        """Map from bounded space to unbounded space using logit."""
        mapped_params = []
        for i, (param, use_sigmoid) in enumerate(zip(params, self.sel)):
            if use_sigmoid:
                lower, upper = self._param_bounds[i]
                # Convert to backend array
                param_t = xp.asarray(param)
                # Logit: log((y - a) / (b - a) / (1 - (y - a) / (b - a)))
                # Simplified: logit((y - a) / (b - a))
                normalized = (param_t - lower) / (upper - lower)
                # Clip to avoid log(0) or log(inf)
                normalized = xp.clip(normalized, 1e-10, 1 - 1e-10)
                logit_val = xp.log(normalized / (1.0 - normalized))
                mapped_params.append(logit_val)
            else:
                mapped_params.append(param)
        return tuple(mapped_params)
    
    @property
    def parameter_names(self):
        return self.base_model.parameter_names
    
    @property
    def free_parameter_names(self):
        return self.base_model.free_parameter_names
    
    def freeze_parameter(self, name):
        self.base_model.freeze_parameter(name)
    
    @property
    def free_parameters_repr(self):
        return tuple([f"sigmoid {p_repr}" if self.sel[self.parameter_names.index(p_name)] else f"{p_repr}"
                      for (p_repr, p_name) in zip(self.base_model.free_parameters_repr, self.free_parameter_names)])
    
    @property
    def parameters_repr(self):
        repr_base_params = self.base_model.parameters_repr
        return tuple([f"sigmoid {p_repr}" for p_repr in repr_base_params])
    
    @property
    def free_parameter_bounds(self):
        # For transformed parameters (sigmoid), bounds are (-inf, inf)
        # For non-transformed parameters, use the base model bounds
        base_bounds = self.base_model.free_parameter_bounds
        mapped_bounds = []
        for i, use_sigmoid in enumerate(self.sel):
            if use_sigmoid:
                mapped_bounds.append((-float('inf'), float('inf')))
            else:
                mapped_bounds.append(base_bounds[i])
        return mapped_bounds
    
    def frozen_copy(self):
        copy = self.copy()
        copy.freeze()
        return copy


class SeparableModel:
    pass


if __name__ == "__main__":
    from rich import print
    from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
    model = SquaredExponentialModel(rho=32)
    print(model)
    lags = xp.array([[0., 0., 0.], [0., 1., 2.]])
    print(model(lags))

    model2 = LogScaleReparameterizedModel(model)
    print(model2.free_parameter_bounds)