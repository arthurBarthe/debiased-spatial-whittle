from .backend import BackendManager
np = BackendManager.get_backend()

# In this file we define covariance models
from abc import ABC, abstractmethod

from autograd.scipy.special import gamma # , kv
from typing import Tuple, List, Dict, Union

class Parameter:
    """
    Class designed to handle parameters of model. Note that a Parameter can have it own core value, or might just
    be a pointer to another parameter. In particular, two parameters might point to a single other parameter.
    In that case, those two parameters always have the same value.
    """
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
        assert self.free and other.free, "Only free parameters (value set to None) can be merged together."
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
        return f'{self.name}: {self.value} ... {self.bounds}'


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
        return [p.value for p in self]

    @property
    def bounds(self):
        return [p.bounds for p in self]

    @property
    def init_guesses(self):
        return [p.init_guess for p in self]

    def free_params(self):
        list_free = list(filter(lambda p: p.free, self))
        return Parameters(list(set(list_free)))

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
        return '\n'.join([p.__repr__() for p in self])


class ParametersUnion(Parameters):
    """A class to represent the union of parameters when using a combination of models
    to define a new model. Handles the case where several of the models are using a common parameter."""

    def __init__(self, parameters: List[Parameters]):
        param_list = []
        for i, params in enumerate(parameters):
            for p in params:
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
        return self.free_params.bounds

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
        pass

    def gradient(self, x: np.ndarray, params: Parameters):
        """Provides the gradient of the covariance functions at the passed lags with respect to
        the passed parameters"""
        gradient = dict([(p.name, np.zeros_like(x[0], dtype=np.float64)) for p in params.param_dict.values()])
        g = self._gradient(x)
        for i, p in enumerate(self.params):
            if p in params.param_dict.values():
                gradient[p.name] += np.take(g, i, axis=-1)
        return gradient

    @abstractmethod
    def _gradient(self, x: np.ndarray):
        pass

    def __setattr__(self, key: str, value: Union[float, Parameter]):
        if 'params' in self.__dict__:
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
        return 'SeparableModel(\n' + '\n'.join([m.__repr__() for m in self.models]) + '\n)'


class ExponentialModel(CovarianceModel):
    def __init__(self):
        sigma = Parameter('sigma', (0.01, 1000))
        rho = Parameter('rho', (0.01, 1000))
        nugget = Parameter('nugget', (1e-6, 1000))
        
        parameters = Parameters([rho, sigma, nugget])
        super(ExponentialModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray, time_domain:bool=False):
        # TODO: time domain
        lags = np.stack(lags, axis=0)
        d = np.sqrt(np.sum(lags**2, axis=0))
        nugget_effect = self.nugget.value*np.all(lags == 0, axis=0)
        
        acf = self.sigma.value**2 * np.exp(- d / self.rho.value) + nugget_effect
    
        if nu is not None:
            acf *= nu/(nu-2)    # t-density covariance
        return acf

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

    def _gradient(self, lags: np.ndarray):
        d = np.abs(lags[self.axis])
        d_rho = (self.sigma.value / self.rho.value) ** 2 * d * np.exp(- d / self.rho.value)
        d_sigma = 2 * self.sigma.value * np.exp(- d / self.rho.value)
        return np.stack((d_rho, d_sigma), axis=-1)

import scipy
from autograd.scipy.special import gamma, iv
from autograd.extend import primitive, defvjp, defjvp
# kv = primitive(scipy.special.kv)
# defvjp(kv, None, lambda ans, n, x: lambda g: -g * (kv(n - 1, x) + kv(n + 1, x)) / 2.0)

def kv_(nu, z):
    if nu % 1 == 0:
        nu +=1e-6
    return (np.pi/2) * (iv(-nu,z) - iv(nu, z)) / np.sin(nu*np.pi)


class MaternModel(CovarianceModel):
    #  TODO: needs more testing!
    def __init__(self):
        rho = Parameter('rho', (0.01, 1000))
        sigma = Parameter('sigma', (0.01, 1000))
        nu = Parameter('nu', (0.01, 1000))
        nugget = Parameter('nugget', (1e-6, 1000))
        
        parameters = Parameters([rho, sigma, nu, nugget])
        super(MaternModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray, time_domain:bool=False, nu:int|None=None):
        
        rho, sigma, v = self.rho.value, self.sigma.value, self.nu.value
        
        if time_domain:
            d = np.sqrt(lags)         # this is the full covariance matrix
            nugget_effect = self.nugget.value*np.eye(len(lags))
        else:
            d = np.sqrt(sum((lag**2 for lag in lags)))
            mask = (d==0)
            nugget_effect = (sigma**2 + self.nugget.value)
        
        # TODO: add specific cases for nu=1/2, 3/2, 5/2, inf
        const = 2 ** (1 - v) / gamma(v)
        args  = np.sqrt(2 * v) * d / rho
        term2 = (args) ** v
        term3 = kv_(v, args + mask)          # TODO: bad solution
        acf = sigma**2 * const * term2 * term3
        acf += (nugget_effect-acf)*mask
        if nu is not None:
            acf *= nu/(nu-2)    # TODO: name collision with t-acf
        return acf
    
    def f(self, freq_grid:list|np.ndarray, infsum_grid:list|np.ndarray, d:int=2):
        '''aliased spectral density, should match with the acf'''
        
        rho, sigma, v = self.rho.value, self.sigma.value, self.nu.value
        
        shape  = freq_grid[0].shape
        d = len(shape)
        N = np.prod(shape)
        
        # TODO: wrong spectral density?
        args   = (np.tile(infsum_grid[i], (N,1,1)) + freq_grid[i].reshape(N,1,1) for i in range(d))
        omega2 = np.sum((arg**2 for arg in args))
        
        term1 = 2**(d) * np.pi**(d/2) * gamma(v + d/2) * (2*v)**v
        term2 = 1 / (gamma(v) * rho**(2*v))
        term3 = ( 2*v/rho**2 + 4*np.pi**2 * omega2 )**(-v - d/2)      
        f = sigma**2 * term1 * term2 * term3
            
        return (np.sum(f, axis=(1,2)).reshape(shape) + self.nugget.value)

    def _gradient(self):
        pass


class SquaredExponentialModel(CovarianceModel):
    def __init__(self):
        rho = Parameter('rho', (0.01, 1000))
        sigma = Parameter('sigma', (0.01, 1000))
        nugget = Parameter('nugget', (1e-6, 1000))
        
        parameters = Parameters([rho, sigma, nugget])
        super(SquaredExponentialModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray, time_domain:bool=False, nu:Union[int, None]=None):
        
        if time_domain:
            d2 = lags         # this is the full covariance matrix
            nugget_effect = self.nugget.value*np.eye(len(lags))
        else:
            d2 = sum((lag**2 for lag in lags))
            nugget_effect = self.nugget.value*np.all(lags == 0, axis=0)
            
        acf = self.sigma.value ** 2 * np.exp(- 0.5*d2 / self.rho.value ** 2) + nugget_effect  # exp(0.5) as well
        
        if nu is not None:
            acf *= nu/(nu-2)    # t-density covariance
        return acf
    
    def f(self, freq_grid:Union[list, np.ndarray], infsum_grid:Union[list, np.ndarray], d:int=2):
        '''aliased spectral density, should match with the acf'''
        
        shape  = freq_grid[0].shape
        N = np.prod(shape)
        
        args   = (np.tile(infsum_grid[i], (N,1,1)) + freq_grid[i].reshape(N,1,1) for i in range(d))
        omega2 = np.sum((arg**2 for arg in args))
        
        # f = sigma2*(2*np.pi*rho**2)**(d/2)*np.exp(-2*(np.pi*rho)**2 * omega2) #+ nugget/(2*np.pi)**2
        f = self.sigma.value**2*self.rho.value**2*(2*np.pi)**(d/2)*np.exp(-.5*(self.rho.value**2*omega2))#/(2*np.pi)**2
        return (np.sum(f, axis=(1,2)).reshape(shape) + self.nugget.value)


    def _gradient(self, lags: np.ndarray):
        """Provides the derivatives of the covariance model evaluated at the passed lags with respect to
        the model's parameters"""
        # TODO: include nugget
        d2 = sum((lag ** 2 for lag in lags))
        d_rho =  2 / self.rho.value ** 3 * d2 * self.sigma.value ** 2 * np.exp(-d2 / self.rho.value ** 2)
        d_sigma = 2 * self.sigma.value * np.exp(- d2 / self.rho.value ** 2)
        return np.stack((d_rho, d_sigma), axis=-1)


# TODO this should not be just a covariance model. Create more general class for model with parameters
class TMultivariateModel(CovarianceModel):
    """
    Model corresponding to a t-multivariate distribution.
    """
    def __init__(self, covariance_model: CovarianceModel):
        self.covariance_model = covariance_model
        nu = Parameter('nu', (1, 1000))
        parameters = ParametersUnion([covariance_model.params, Parameters([nu, ])])
        super(TMultivariateModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        acv_gaussian = self.covariance_model(lags)
        return acv_gaussian * self.nu_1.value / (self.nu_1.value - 2)

    def _gradient(self, lags: np.ndarray):
        return self.covariance_model._gradient(lags) * self.nu.value / (self.nu.value - 2)


class SquaredModel(CovarianceModel):
    """
    Covariance model for a process defined pointwise as the square of a Gaussian process
    """
    def __init__(self, latent_model: CovarianceModel):
        self.latent_model = latent_model
        super(SquaredModel, self).__init__(self.latent_model.params)

    def __call__(self, lags: np.ndarray):
        return 2 * self.latent_model(lags) ** 2 + self.latent_model.sigma ** 2

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class ChiSquaredModel(CovarianceModel):
    """
    Covariance model for a Chi-Squared random field
    """
    def __init__(self, latent_model: CovarianceModel):
        self.latent_model = latent_model
        dof = Parameter('dof', (1, 1000))
        super(ChiSquaredModel, self).__init__(ParametersUnion([self.latent_model.params, Parameters([dof, ])]))

    def __call__(self, lags: np.ndarray):
        return (self.dof_1.value * (2 * self.latent_model(lags) ** 2 + self.latent_model.sigma ** 2)
        + self.dof_1.value * (self.dof_1.value - 1) * self.latent_model.sigma ** 2)

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class PolynomialModel(CovarianceModel):
    """
    Covariance model for a process defined pointwise via a polynomial function
    applied to a latent Gaussian process.
    Currently limited to order 2.
    """

    def __init__(self, latent_model: CovarianceModel):
        self.latent_model = latent_model
        a = Parameter('a', (-1, 1))
        b = Parameter('b', (-1, 1))
        pol_parameters = Parameters([a, b])
        super(PolynomialModel, self).__init__(ParametersUnion([self.latent_model.params, pol_parameters]))

    def __call__(self, lags: np.ndarray):
        a = self.a_1.value
        b = self.b_1.value
        term1 = 2 * self.latent_model(lags) ** 2 + self.latent_model.sigma ** 2
        term2 = self.latent_model(lags)
        return a**2 * term1 + b**2 * term2

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


class MaternCovarianceModel(CovarianceModel):
    def __init__(self):
        sigma = Parameter('sigma', (0.01, 1000))
        rho = Parameter('rho', (0.01, 1000))
        nu = Parameter('nu', (0.5, 100))
        parameters = Parameters([sigma, nu, rho])
        super(MaternCovarianceModel, self).__init__(parameters)

    def __call__(self, lags: np.ndarray):
        lags = np.stack(lags, axis=0)
        d = np.sqrt(np.sum(lags ** 2, axis=0))
        sigma, rho, nu = self.sigma.value, self.rho.value, self.nu.value
        if nu==1.5:
            K = np.sqrt(3) * d / rho
            return (1.0 + K) * np.exp(-K) * sigma**2
        term1 = 2 ** (1 - nu) / gamma(nu)
        term2 = (np.sqrt(2 * nu) * d / rho) ** nu
        term3 = kv_(nu, np.sqrt(2 * nu) * d / rho)
        val = sigma ** 2 * term1 * term2 * term3
        val[d == 0] = sigma ** 2
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
    g = model.gradient(g.lags_unique, Parameters([model.rho, ]))['rho']
    g2 = (acv2 - acv1) / epsilon
    assert_allclose(g, g2, rtol=1e-3)