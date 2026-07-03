from abc import ABC, abstractmethod
from debiased_spatial_whittle.backend import BackendManager
xp = BackendManager.get_backend()
inv = BackendManager.get_inv()

from torch.autograd.functional import jacobian


class ModelParameter:
    def __init__(self, default, bounds=(None, None), doc=""):
        self.default = xp.asarray(default).astype(xp.float64)
        self.bounds = bounds
        self.doc = doc

    def __set_name__(self, owner, name):
        self.name = name
        self.long_name = f'{owner.__name__}_{name}'
        if not hasattr(owner, '_parameters'):
            owner._parameters = []
        if not hasattr(owner, '_parameter_bounds'):
            owner._parameter_bounds = dict()
        owner._parameters.append(name)
        owner._parameter_bounds[self.name] = self.bounds

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(f"_{self.name}", self.default)

    def __set__(self, obj, value):
        if hasattr(obj, '_frozen_parameters') and self.name in obj._frozen_parameters:
            raise ValueError(f"Parameter {self.name} is frozen and cannot be set.")
        if value is not None:
            obj.__dict__[f"_{self.name}"] = xp.asarray(value).astype(xp.float64)



class ModelInterface(ABC):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @name.setter
    @abstractmethod
    def name(self, value):
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

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other):
        raise NotImplementedError()




class CovarianceModel(ModelInterface):

    def __init__(self, children: tuple[ModelInterface], *params, name: str = None):
        self.name = name
        self._frozen_parameters = []
        self.children = children
        self.assign_params(*params)

    def assign_params(self, *params):
        for param_name, param_value in zip(self._parameters, params):
            setattr(self, param_name, param_value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value if value else self.__class__.__name__

    @property
    def free_parameter_bounds(self):
        out = []
        for param_name in self._parameters:
            if not param_name in self._frozen_parameters:
                out.append(self._parameter_bounds[param_name])
        for child in self.children:
            out.extend(child.free_parameter_bounds)
        return out

    # old method name
    def free_parameter_bounds_to_list_deep(self):
        return self.free_parameter_bounds

    def set_parameter_bounds(self, name: str, bounds: tuple[float, float]) -> None:
        model_name, param_name = name.split("_")
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
    def parameter_names(self) -> tuple:
        out = []
        for param_name in self._parameters:
            out.append(f'{self.name}_{param_name}')
        for child in self.children:
            out.extend(child.parameter_names)
        return out

    @property
    def free_parameter_names(self) -> tuple:
        out = []
        for param_name in self._parameters:
            if not param_name in self._frozen_parameters:
                out.append(f'{self.name}_{param_name}')
        for child in self.children:
            out.extend(child.parameter_names)
        return out

    def get_parameter(self, name: str):
        model_name, param_name = name.split("_")
        if model_name == self.name:
            return getattr(self, param_name)
        else:
            for child in self.children:
                value = child.get_parameter(name)
                if value is not None:
                    return value
            return None

    def set_parameter(self, name, value) -> bool:
        model_name, param_name = name.split("_")
        if model_name == self.name:
            setattr(self, param_name, value)
            return True
        else:
            for child in self.children:
                value = child.set_parameter(name, value)
                if value:
                    return True
            return False

    def freeze_parameter(self, name):
        model_name, param_name = name.split("_")
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
                is_last_param = (idx == len(self._parameters) - 1) and (not self.children)
                param_connector = "└── " if is_last_param else "├── "
                lines.append(f"{new_prefix}{param_connector}{param_name}: {param_value}{fixed_marker}")
        
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
            html.append('<tr><th style="text-align:left;">Parameter</th><th style="text-align:left;">Value</th><th>Fixed</th></tr>')
            for param_name in self._parameters:
                param_obj = getattr(self.__class__, param_name)
                param_value = getattr(self, param_name)
                is_fixed = param_name in self._frozen_parameters
                fixed_str = 'Yes' if is_fixed else 'No'
                fixed_style = 'color:orange;' if is_fixed else ''
                html.append(f'<tr><td>{param_name}</td><td>{param_value}</td><td style="{fixed_style}">{fixed_str}</td></tr>')
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


class ReparameterizedModel(ModelInterface, ABC):
    """
    Class that allows to use an alternative parameterization of a base model.
    """
    def __init__(self, base_model: ModelInterface, name: str = None):
        self.base_model = base_model
        self._parameter_bounds = []
        self._frozen_parameters = []
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        name = name if name else self.__class__.__name__
        self._name = name

    def map_parameters(self, *params):
        raise NotImplementedError()

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
        new_parameters = [current_parameters[i] if name != pname else value for i, pname in enumerate(self.free_parameter_names)]
        self.base_model.set_parameters(dict(zip(self.base_model.parameter_names, self.map_parameters(new_parameters))))

    def set_parameter_bounds(self, name: str, bounds: tuple[float, float]) -> None:
        self._parameter_bounds.append(bounds)

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

    def freeze_parameter(self, name: str):
        self._frozen_parameters.append(name)

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        base_model_params = self.map_parameters(*params)
        return self.base_model.compute(lags, *base_model_params)

    def __add__(self, other):
        return SumModel(self, other)

    def __mul__(self, other):
        return ProductModel(self, other)


class LogScaleReparameterizedModel(ReparameterizedModel):
    """
    Class that allows to use a log scale parameterization of a base model.
    """
    def __init__(self, base_model: ModelInterface):
        super().__init__(base_model)

    def map_parameters(self, *params):
        params_array = xp.stack(params)
        mapped_params = xp.exp(params_array)
        return xp.split(mapped_params, 1)

    def imap_parameters(self, *params):
        params_array = xp.stack(params)
        mapped_params = xp.log(params_array)
        return xp.split(mapped_params, 1)

    @property
    def parameter_names(self):
        return self.base_model.parameter_names

    @property
    def free_parameter_names(self):
        return self.base_model.free_parameter_names


class SeparableModel:
    pass


if __name__ == "__main__":
    from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
    model = SquaredExponentialModel(rho=32)
    mm = LogScaleReparameterizedModel(model)
    print(mm.parameters)
    lags = xp.array([[0., 0., 0.], [0., 1., 2.]])
    print(model(lags))
    print(mm(lags))
    print(model.jacobian(lags))
    print(mm.jacobian(lags))