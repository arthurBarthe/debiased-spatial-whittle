from debiased_spatial_whittle.backend import BackendManager
xp = BackendManager.get_backend()

from torch.autograd.functional import jacobian


class ModelParameter:
    def __init__(self, default, bounds=(None, None), doc=""):
        self.default = default
        self.bounds = bounds
        self.doc = doc

    def __set_name__(self, owner, name):
        self.name = name
        self.long_name = f'{owner.__name__}_{name}'
        if not hasattr(owner, '_parameters'):
            owner._parameters = []
        owner._parameters.append(name)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(f"_{self.name}", self.default)

    def __set__(self, obj, value):
        if hasattr(obj, '_frozen_parameters') and self.name in obj._frozen_parameters:
            raise ValueError(f"Parameter {self.name} is frozen and cannot be set.")
        if value is not None:
            obj.__dict__[f"_{self.name}"] = xp.asarray(value).astype(xp.float64)



class ModelInterface:
    @property
    def name(self):
        raise NotImplementedError()

    @name.setter
    def name(self, value):
        raise NotImplementedError()

    def _build_name(self, name):
        raise NotImplementedError()

    @property
    def parameters(self) -> tuple:
        raise NotImplementedError()

    @property
    def n_parameters(self):
        return len(self.parameters)

    @property
    def free_parameters(self) -> tuple:
        raise NotImplementedError()

    @property
    def n_free_parameters(self):
        return len(self.free_parameters)

    def free_parameter_bounds(self):
        raise NotImplementedError()

    @property
    def parameter_names(self) -> tuple:
        raise NotImplementedError()

    @property
    def free_parameter_names(self) -> tuple:
        raise NotImplementedError()

    def get_parameter(self, name: str):
        raise NotImplementedError()

    def set_parameter(self, name, value) -> bool:
        raise NotImplementedError()

    def get_parameters(self, names: list[str]) -> tuple:
        return tuple([self.get_parameter(name) for name in names])

    def set_parameters(self, name_values: dict[str, object]):
        for param_name, param_value in name_values.items():
            self.set_parameter(param_name, param_value)

    # methods useful for optimizers --------------------

    def update_free_parameters(self, values):
        free_parameter_names = self.free_parameter_names
        self.set_parameters(dict(zip(free_parameter_names, values)))

    def free_parameter_values_to_array_deep(self):
        return xp.array(self.free_parameters)

    # --------------------------------------------------

    def freeze_parameter(self, name, value) -> bool:
        raise NotImplementedError()

    def compute(self, lags: xp.ndarray, *params) -> xp.ndarray:
        """
        Here we expect the model parameters to be passed via param_args.
        param_args: can be a tuple (should have size n_parameters) or a named tuple (not implemented yet)
        """
        raise NotImplementedError()

    def __call__(self, lags: xp.ndarray) -> xp.ndarray:
        lags = xp.asarray(lags)
        params = self.parameters
        return self.compute(lags, *params)

    def jacobian(self, lags: xp.ndarray, param_names: tuple[str] = None) -> xp.ndarray:
        """Obtain the jacobian of covariance values at lags with respect to the passed parameters"""
        raise NotImplementedError()




class CovarianceModel(ModelInterface):

    def __init__(self, children: tuple[ModelInterface], *params, name: str = None):
        self.name = self._build_name(name)
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
        self._name = value

    def _build_name(self, name):
        return name if name else self.__class__.__name__

    @property
    def parameters(self) -> tuple:
        out = []
        for param_name in self._parameters:
            out.append(getattr(self, param_name))
        for child in self.children:
            out.extend(child.parameters)
        return tuple(out)

    @property
    def free_parameters(self) -> tuple:
        out = []
        for param_name in self._parameters:
            if not param_name in self._frozen_parameters:
                out.append(getattr(self, param_name))
        for child in self.children:
            out.extend(child.parameters)
        return tuple(out)

    def free_parameter_bounds(self):
        out = []
        for param_name in self._parameters:
            if not param_name in self._frozen_parameters:
                out.append(getattr(self.__class__, param_name).bounds)
        for child in self.children:
            out.extend(child.free_parameter_bounds())

    # old method name
    def free_parameter_bounds_to_list_deep(self):
        return self.free_parameter_bounds()

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

    def jacobian(self, lags: xp.ndarray, param_names: tuple[str] = None) -> xp.ndarray:
        if param_names is None:
            param_names = self.parameter_names
        param_values = self.get_parameters(param_names)
        out = jacobian(lambda *args: self.compute(lags, *args), param_values, strategy="forward-mode", vectorize=True)
        return dict(zip(param_names, out))

    def _split_children_params(self, *params):
        out = []
        for child in self.children:
            n_params = child.n_parameters
            temp, params = params[:n_params], params[n_params:]
            out.append(temp)

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


class BaseCovarianceModel(CovarianceModel):
    def __init__(self, *params, name=None):
        super().__init__((), *params, name=name)