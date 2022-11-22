# In this file we define covariance models
from abc import ABC, abstractmethod
import numpy as np
from numpy.fft import fft, fftn, ifftn, fftshift, ifftshift
from scipy.optimize import fmin_l_bfgs_b
import warnings
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict

from expected_periodogram import compute_ep


class Parameter:
    def __init__(self, name: str, bounds: Tuple[float, float]):
        #TODO hide point_to
        #TODO add registry to ensure no duplicate names? Add access to registry by name of parameter?
        self.point_to = None
        self.name = name
        self.bounds = bounds
        self.value = None
        self.init_guess = 1.

    #TODO add property name and make it point to if adequate

    @property
    def value(self):
        if self.point_to is not None:
            return self.point_to.value
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
        assert self.free and other.free
        new_name = self.name + ' and ' + other.name
        new_min = max(self.bounds[0], other.bounds[0])
        new_max = min(self.bounds[1], other.bounds[1])
        new_bounds = (new_min, new_max)
        new_parameter = Parameter(new_name, new_bounds)
        self.point_to = new_parameter
        other.point_to = new_parameter
        return new_parameter

    def __pow__(self, power, modulo=None):
        return self.value ** power

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
        return f'{self.name}: {self.value}'


class Parameters:
    """Wrapper for a dictionary of parameters. The keys are the names of the parameters in the model, but two
    different keys could point to a single Parameter object"""

    def __init__(self, parameters: List[Parameter]):
        #TODO make param_dict hidden
        self.param_dict = dict([(p.name, p) for p in parameters])

    @property
    def names(self):
        return list(self.param_dict.keys())

    @property
    def values(self):
        return [p.value for p in self.param_dict.values()]

    @property
    def bounds(self):
        return [p.bounds for p in self.param_dict.values()]

    @property
    def init_guesses(self):
        return [p.init_guess for p in self.param_dict.values()]

    def free_params(self):
        list_free = list(filter(lambda p: p.free, self.param_dict.values()))
        return Parameters(list(set(list_free)))

    def __getitem__(self, item):
        return self.param_dict[item]
    # TODO add some checks that no two parameters have the same name

    def __setitem__(self, key, value):
        self.param_dict[key] = value

    def __len__(self):
        # TODO change to return the number of unique parameters?
        return len(self.param_dict.keys())

    def update_values(self, updates: Dict[str, float]):
        for p_name, value in updates.items():
            self[p_name].value = value

    def __repr__(self):
        return '\n'.join([p.__repr__() for p in self.param_dict.values()])


class ParametersUnion(Parameters):
    """A class to represent the union of parameters when using a combination of models
    to define a new model. Handles the case where several of the models are using a common parameter."""

    def __init__(self, parameters: List[Parameters]):
        param_list = []
        for i, params in enumerate(parameters):
            # TODO line below is ugly add property
            for p in params.param_dict.values():
                p.name = p.name + '_' + str(i)
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
    def n_params(self):
        """Number of parameters of the model"""
        return len(self.params)

    @property
    def param_names(self):
        """Names of the parameters of the model"""
        return self.params.names

    @property
    def param_values(self):
        return self.params.values

    @property
    def free_params(self):
        return self.params.free_params()

    @property
    def free_param_bounds(self):
        free_params = self.free_params
        return free_params.bounds

    def update_free_params(self, updates):
        self.params.update(updates)

    def merge_parameters(self, param_names: Tuple[str, str]):
        """Merges two parameters into one. This can only be done for two free parameters
        as for fixed parameters this would not make sense except in the trivial case where
        the two parameters are already equal."""
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
        pass

    def gradient(self, x: np.ndarray, params: Parameters):
        """Provides the gradient of the covariance functions at the passed lags with respect to
        the passed parameters"""
        gradient = dict([(p.name, np.zeros_like(x[0], dtype=np.float64)) for p in params.param_dict.values()])
        g = self._gradient(x)
        #TODO ugly line (also above and again further below)
        for i, p in enumerate(self.params.param_dict.values()):
            if p in params.param_dict.values():
                gradient[p.name] += np.take(g, i, axis=-1)
        return gradient

    #@abstractmethod
    def _gradient(self, x: np.ndarray):
        pass

    def __setattr__(self, key, value):
        if 'params' in self.__dict__:
            if key in self.param_names:
                self.params[key].value = value
                return
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.param_names:
            return self.params[item]

    def __repr__(self):
        return self.name + '(\n' + self.params.__repr__() + '\n)'


class SeparableModelOld(CovarianceModel):
    """Class for a separable covariance model based on a list of covariance models"""
    # TODO only works in dimension 2 right now

    def __init__(self, models: List[CovarianceModel]):
        self.models = models
        parameters = ParametersUnion([m.params for m in models])
        super().__init__(parameters)

    def __call__(self, lags: np.ndarray):
        cov_seqs = [model_i([lags_i, ]) for model_i, lags_i in zip(self.models, lags)]
        return cov_seqs[0] * cov_seqs[1]

    def _gradient(self, lags):
        covs = [model_i([lags_i, ]) for model_i, lags_i in zip(self.models, lags)]
        gradients = [model_i._gradient([lags_i, ]) for model_i, lags_i in zip(self.models, lags)]
        g1 = gradients[0] * np.expand_dims(covs[1], axis=-1)
        g2 = gradients[1] * np.expand_dims(covs[0], axis=-1)
        return np.concatenate((g1, g2), axis=-1)

    def __repr__(self):
        return 'SeparableModel(\n' + '\n'.join([m.__repr__() for m in self.models]) + '\n)'


class SeparableModel(CovarianceModel):
    """Class for a separable covariance model based on a list of covariance models"""
    #TODO only works in dimension 2 right now
    #TODO we should allow for a semi-separable model: for instance, time and space are separable, but 2d space is not

    def __init__(self, models: List[CovarianceModel]):
        self.models = models
        parameters = ParametersUnion([m.params for m in models])
        super(SeparableModel, self).__init__(parameters)

    def __call__(self, lags: List[np.ndarray]):
        model1, model2 = self.models
        cov1 = model1(lags)
        cov2 = model2(lags)
        return cov1 * cov2

    def _gradient(self, lags):
        covs = [model_i(lags) for model_i in self.models]
        gradients = [model_i._gradient(lags) for model_i in self.models]
        g1 = gradients[0] * np.expand_dims(covs[1], axis=-1)
        g2 = gradients[1] * np.expand_dims(covs[0], axis=-1)
        return np.concatenate((g1, g2), axis=-1)

    def __repr__(self):
        return 'SeparableModel(\n' + '\n'.join([m.__repr__() for m in self.models]) + '\n)'


class ExponentialModel(CovarianceModel):
    def __init__(self):
        sigma = Parameter('sigma', (0.01, 1000))
        rho = Parameter('rho', (0.01, 1000))
        parameters = Parameters([rho, sigma])
        super(ExponentialModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        d = np.sqrt(sum((lag**2 for lag in lags)))
        return self.sigma.value**2 * np.exp(- d / self.rho.value)

    def _gradient(self, lags: np.ndarray):
        """Provides the derivatives of the covariance model evaluated at the passed lags with respect to
        the model's parameters"""
        d = np.sqrt(sum((lag ** 2 for lag in lags)))
        d_rho = (self.sigma.value / self.rho.value) ** 2 * d * np.exp(- d / self.rho.value)
        d_sigma = 2 * self.sigma.value * np.exp(- d / self.rho.value)
        return np.stack((d_rho, d_sigma), axis=-1)


class ExponentialModelUniDirectional(CovarianceModel):
    """Class for the implementation of a Unidirectional covariance model. At the moment, we only implement
    the case where the direction is aligned with one of the axis. """
    def __init__(self, axis: int):
        sigma = Parameter('sigma', (0.01, 1000))
        rho = Parameter('rho', (0.01, 1000))
        parameters = Parameters([rho, sigma])
        super(ExponentialModelUniDirectional, self).__init__(parameters)
        self.axis = axis
        # TODO add parameter for orientation

    def __call__(self, lags: List[np.ndarray]):
        d = np.abs(lags[self.axis])
        return self.sigma.value ** 2 * np.exp(- d / self.rho.value)



if __name__ == '__main__':
    m1 = ExponentialModelUniDirectional(axis=0)
    m1.rho = 8
    m1.sigma = 1
    m2 = ExponentialModelUniDirectional(axis=1)
    m2.rho = 32
    m2.sigma = 2
    model = SeparableModel((m1, m2))

    g = RectangularGrid((128, 128))
    sampler = SamplerOnRectangularGrid(model, g)

    z = sampler()
    plt.figure()
    plt.imshow(z, cmap='Spectral')
    plt.show()