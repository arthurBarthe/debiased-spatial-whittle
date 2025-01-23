import numpy as np
import param
from param import Parameterized
from abc import ABC, abstractmethod


class ModelParameter(param.Parameter):
    __slots__ = [
        "bounds",
    ]

    def __init__(self, *args, **kwargs):
        self.bounds = kwargs.pop("bounds")
        super().__init__(*args, allow_refs=True, per_instance=True, **kwargs)

    @property
    def free(self) -> bool:
        """true ii the model parameter is free, i.e. not readonly and not constant"""
        return not (self.readonly or self.constant)

    def fix_to_current_value(self):
        """fix the parameter to its current value"""
        self.readonly = True
        self.constant = True


class ModelInterface:
    """
    Class defining the general interface for covariance models. This should remain implementation-independent.

    Attributes
    ----------
    parameters
        dictionary of (parameter name, parameter object) pairs

    free_parameters
        list of free parameters

    n_free_parameters: int
        number of free parameters
    """

    @property
    def parameters(self) -> dict[str, ModelParameter]:
        pass

    @property
    def free_parameters(self) -> list[str]:
        pass

    @property
    def n_free_parameters(self) -> int:
        pass

    def __call__(self, lags: np.ndarray):
        """
        Evaluate covariance model at passed array of lags.

        Parameters
        ----------
        lags
            Array of lags. Shap (ndim, n1, ..., nk)

        Returns
        -------
        cov
            covariance values.

            Shape (n1, ..., nk) for univariate data

            Shape (n1, ..., nk, p, p) for p-variate data, p > 1.
        """
        pass

    def free_parameter_values_to_array(self) -> np.ndarray:
        """
        Provide the values of the model's free parameters as an array.
        Useful for use by numerical optimizer.

        Returns
        -------
        param_values
            array of free parameter values
        """
        pass

    def update_free_parameters_from_array(self, param_array: np.ndarray):
        """
        Update the model's free parameters from an array of values.
        Useful for use by numerical optimizers.

        Parameters
        ----------
        param_array
            array of parameter values

        Returns
        -------

        """
        pass

    def free_parameter_bounds_to_list(self):
        """
        Provide the bounds of the model's free parameters as a list.
        Useful for use in numerical optimizers.

        Returns
        -------
        bounds: list[tuple[float, float]]
            list of min, max bounds for the model's free parameters
        """

    def gradient(self, lags: np.ndarray, params: list[ModelParameter]):
        """
        Compute the gradient of the model with respect to the passed parameters

        Parameters
        ----------
        lags


        params
            parameters for which we require the derivative

        Returns
        -------

        """


class BaseModel(Parameterized, ModelInterface):
    """
    Class to define low-level covariance models (e.g. exponential, squared exponential).
    """

    @property
    def parameters(self):
        return self.param.objects()

    @property
    def free_parameters(self) -> list[str]:
        out = []
        for p in self.parameters.values():
            if isinstance(p, ModelParameter) and p.free:
                out.append(p.name)
        return out

    @property
    def n_free_parameters(self):
        return len(self.free_parameters)

    def update_free_parameters_from_array(self, param_values: np.ndarray):
        """In the case of a simple model, we simply update the free parameters"""
        a = param_values[: self.n_free_parameters]
        for p_name, value in zip(self.free_parameters, a):
            setattr(self, p_name, value)

    def free_parameter_values_to_array(self):
        list_values = []
        for p in self.free_parameters:
            list_values.append(getattr(self, p))
        return np.array(list_values)

    def free_parameter_bounds_to_list(self):
        list_bounds = []
        for p in self.free_parameters:
            list_bounds.append(getattr(self.param, p).bounds)
        return list_bounds

    def __add__(self, other):
        return SumModel([self, other])


class CompoundModel(ModelInterface, Parameterized):
    """
    Class to define a covariance model from a combination of other covariance models

    Attributes
    ----------
    children: list[ModelInterface]
        list of child covariance models
    """

    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children

    @property
    def n_free_parameters(self):
        out = self._count_own_free_parameters()
        for child in self.children:
            out += child.n_free_parameters
        return out

    @property
    def free_parameters(self) -> list[str]:
        out = self._get_own_free_parameters()
        for child in self.children:
            out.extend(child.free_parameters)
        return out

    def update_free_parameters_from_array(self, param_values):
        n_own_free = self._count_own_free_parameters()
        a, b = param_values[:n_own_free], param_values[n_own_free:]
        for p_name, value in zip(self._get_own_free_parameters(), a):
            setattr(self, p_name, value)
        # update parameters of children
        for child in self.children:
            child.update_free_parameters_from_array(b)
            b = b[child.n_free_parameters :]

    def free_parameter_values_to_array(self):
        list_values = []
        for p in self._get_own_free_parameters():
            list_values.append(getattr(self, p))
        array_values = np.array(list_values)
        return np.concatenate(
            [
                array_values,
            ]
            + [child.free_parameter_values_to_array() for child in self.children]
        )

    def free_parameter_bounds_to_list(self):
        list_bounds = []
        for p in self._get_own_free_parameters():
            list_bounds.append(getattr(self.param, p).bounds)
        for child in self.children:
            list_bounds.extend(child.free_parameter_bounds_to_list())
        return list_bounds

    def _get_own_free_parameters(self) -> list[str]:
        out = []
        for p in self.param.objects().values():
            if isinstance(p, ModelParameter) and p.free:
                out.append(p.name)
        return out

    def _count_own_free_parameters(self):
        return len(self._get_own_free_parameters())


class ExponentialModel(BaseModel):
    rho = ModelParameter(default=1.0, bounds=(0, None), doc="Range parameter")
    sigma = ModelParameter(default=1.0, bounds=(0, 1), doc="Amplitude parameter")

    def __call__(self, lags: np.ndarray):
        d = np.sqrt(np.sum(lags**2, 0)) / self.rho
        return self.sigma**2 * np.exp(-d)


class SquaredExponentialModel(BaseModel):
    rho = ModelParameter(default=1.0, bounds=(0, None), doc="Range parameter")
    sigma = ModelParameter(default=1.0, bounds=(0, 1), doc="Amplitude parameter")

    def __call__(self, lags: np.ndarray):
        d = np.sum(lags**2, 0) / (2 * self.rho**2)
        return self.sigma**2 * np.exp(-d)


class SumModel(CompoundModel):
    """Class that allows to define a new model as the sum of several models."""

    sigma = ModelParameter(default=1.0, bounds=(0, None))

    def __init__(self, children, *args, **kwargs):
        super().__init__(children, *args, **kwargs)

    def __call__(self, lags: np.ndarray):
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

    sigma = ModelParameter(default=1.0, bounds=(0, None), doc="Amplitude")
    nugget = ModelParameter(default=0.0, bounds=(0, 1), doc="Nugget amplitude")

    def __init__(self, model, *args, **kwargs):
        super().__init__(
            [
                model,
            ],
            *args,
            **kwargs,
        )

    def __call__(self, lags: np.ndarray):
        return (
            np.all(lags == 0, 0) * self.nugget
            + (1 - self.nugget) * self.children[0](lags)
        ) * self.sigma**2
