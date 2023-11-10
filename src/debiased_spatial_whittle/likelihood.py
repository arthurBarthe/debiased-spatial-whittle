import warnings

from .backend import BackendManager
np = BackendManager.get_backend()
import numpy

from .samples import SampleOnRectangularGrid
from .simulation import SamplerOnRectangularGrid


from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.signal.windows import hann as hanning
from .periodogram import compute_ep_old
from .confidence import CovarianceFFT, McmcDiags


fftn = np.fft.fftn
inv = np.linalg.inv
zeros = BackendManager.get_zeros()

def prod_list(l):
    if len(l) == 0:
        return 1
    else:
        return l[0] * prod_list(l[1:])


def shape_two(shape):
    new_shape = []
    for e in shape:
        new_shape.append(2 * e)
    return tuple(new_shape)


def periodogram(y, grid=None, fold=True):
    shape = y.shape
    if grid is None:
        n = prod_list(shape)
    else:
        n = np.sum(grid**2)
    if not fold:
        shape = shape_two(shape)
    return 1 / n * abs(fftn(y, shape))**2


def apply_taper(y):
    n_1, n_2 = y.shape
    taper_1 = hanning(n_1).reshape((n_1, 1))
    taper_2 = hanning(n_2).reshape((1, n_2))
    taper = np.dot(taper_1, taper_2)
    return y * taper, taper


def whittle(per, e_per):
    n = prod_list(per.shape)
    return 1 / n * np.sum(np.log(e_per) + per / e_per)


def whittle_prime(per, e_per, e_per_prime):
    n = prod_list(per.shape)
    if e_per.ndim != e_per_prime.ndim:
        out = []
        for i in range(e_per_prime.shape[-1]):
            out.append(whittle_prime(per, e_per, e_per_prime[..., i]))
        return out
    return 1 / n * np.sum((e_per - per) * e_per_prime / e_per ** 2)


def fit(y, grid, cov_func, init_guess, fold=True, cov_func_prime=None, taper=False, opt_callback=None):
    # apply taper if required
    if taper:
        y, taper = apply_taper(y)
        grid = grid * taper

    per = periodogram(y, grid, fold=fold)

    if cov_func_prime is not None:
        def opt_func(x):
            e_per = compute_ep_old(lambda lags: cov_func(lags, *x), grid, fold=fold)
            e_per_prime = compute_ep_old(lambda lags: (cov_func_prime(lags, *x))[0], grid, fold=fold)
            return whittle(per, e_per), whittle_prime(per, e_per, e_per_prime)

        est_, fval, info = fmin_l_bfgs_b(lambda x: opt_func(x),
                                         init_guess,
                                         maxiter=100,
                                         bounds=[(0.01, 1000), ] * len(init_guess),
                                         callback=opt_callback)
    else:
        def opt_func(x):
            e_per = compute_ep_old(lambda lags: cov_func(lags, *x), grid, fold=fold)
            return whittle(per, e_per)
        est_, fval, info = fmin_l_bfgs_b(lambda x: opt_func(x),
                                         init_guess,
                                         maxiter=100,
                                         approx_grad=True,
                                         bounds=[(0.01, 1000), ] * len(init_guess),
                                         callback=opt_callback)

    if info['warnflag'] != 0:
        warnings.warn('Issue during optimization')

    return est_



#########NEW OOP version
from .periodogram import Periodogram, ExpectedPeriodogram
from .models import CovarianceModel, Parameters
from typing import Callable, Union


def whittle_prime(per, e_per, e_per_prime):
    n = prod_list(per.shape)
    if e_per.ndim != e_per_prime.ndim and e_per_prime.shape[-1] > 1:
        out = []
        for i in range(e_per_prime.shape[-1]):
            out.append(whittle_prime(per, e_per, e_per_prime[..., i]))
        return np.stack(out, axis=-1)
    return 1 / n * np.sum((e_per - per) * e_per_prime / e_per ** 2)


class DebiasedWhittle:
    def __init__(self, periodogram: Periodogram, expected_periodogram: ExpectedPeriodogram):
        self.periodogram = periodogram
        self.expected_periodogram = expected_periodogram
        self.frequency_mask = None

    @property
    def frequency_mask(self):
        if self._frequency_mask is None:
            return 1
        else:
            return self._frequency_mask

    @frequency_mask.setter
    def frequency_mask(self, value: np.ndarray):
        """
        Define a mask in the spectral domain to fit only certain frequencies

        Parameters
        ----------
        value
            mask of zeros and ones
        """
        if value is not None:
            assert value.shape == self.expected_periodogram.grid.n, "shape mismatch between mask and grid"
        self._frequency_mask = value

    def whittle(self, p: np.ndarray, ep: np.ndarray):
        """
        Compute the Whittle distance between a periodogram p and a spectral density function ep

        Parameters
        ----------
        p
            periodogram of the data
        ep
            spectral density
        Returns
        -------

        """
        n_points = self.expected_periodogram.grid.n_points
        return 1 / n_points * np.sum((np.log(ep) + p / ep) * self.frequency_mask)

    def __call__(self, z: np.ndarray, model: CovarianceModel, params_for_gradient: Parameters = None):
        # TODO add a class sample which contains the data and the grid?
        """Computes the likelihood for this data"""
        p = self.periodogram(z)                # you are recomputing I for each iteration i think
        ep = self.expected_periodogram(model)
        whittle = self.whittle(p, ep)
        if BackendManager.backend_name == 'torch':
            whittle = whittle.item()
        if not params_for_gradient:
            return whittle
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        d_whittle = whittle_prime(p, ep, d_ep)
        if BackendManager.backend_name == 'torch':
            d_whittle = d_whittle.cpu().numpy()
        return whittle, d_whittle

    def expected(self, true_model: CovarianceModel, eval_model: CovarianceModel):
        """
        Evaluate the expectation of the Debiased Whittle likelihood estimator for a given
        parameter.

        Parameters
        ----------
        true_model
            Covariance model of the process
        eval_model
            Covariance model for which we evaluate the likelihood

        Returns
        -------
            Expectation of the Debiased Whittle likelihood under true_model, evaluated at
            eval_model
        """
        ep_true = self.expected_periodogram(true_model)
        ep_eval = self.expected_periodogram(eval_model)
        return np.sum(np.log(ep_eval) + ep_true / ep_eval)

    def fisher(self, model: CovarianceModel, params_for_gradient: Parameters):
        """Provides the expectation of the hessian matrix"""
        ep = self.expected_periodogram(model)
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        h = zeros((len(params_for_gradient), len(params_for_gradient)))
        for i1, p1_name in enumerate(params_for_gradient.names):
            for i2, p2_name in enumerate(params_for_gradient.names):
                d_ep1 = d_ep[..., i1]
                d_ep2 = d_ep[..., i2]
                h[i1, i2] = np.sum(d_ep1 * d_ep2 / ep**2)
        # TODO ugly
        return h / self.expected_periodogram.grid.n_points

    def jmatrix(self, model: CovarianceModel, params_for_gradient: Parameters, mcmc_mode: bool = False):
        """
        Provides the variance matrix of the score (gradient of likelihood) under the specified model.

        Parameters
        ----------
        model
            Covariance model
        params_for_gradient
            Parameters with respect to which we take the derivative
        mcmc_mode
            Whether we use mcmc approximation
        Returns
        -------
        np.ndarray
            The predicted covariance matrix of the score, with parameters ordered according to params_for_gradient
        """
        n_params = len(params_for_gradient)
        jmat = np.zeros((n_params, n_params))
        grid = self.expected_periodogram.grid
        n1, n2 = grid.n
        covariance_fft = CovarianceFFT(grid)
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        ep = self.expected_periodogram(model)

        for i in range(n_params):
            for j in range(n_params):
                # TODO get rid of repeated computations for efficiency
                d_epi = np.take(d_ep, i, -1)
                d_epj = np.take(d_ep, j, -1)
                if not mcmc_mode:
                    s1 = covariance_fft.exact_summation1(model, self.expected_periodogram, d_epi / ep**2, d_epj / ep**2)
                else:
                    mcmc = McmcDiags(model, self.expected_periodogram, d_epi / ep, d_epj / ep)
                    mcmc.run(500)
                    s1 = mcmc.estimate()
                #s2 = covariance_fft.exact_summation2(model, self.expected_periodogram, d_epi/ ep**2, d_epj / ep**2)
                s2 = s1
                print(f'{s1=}, {s2=}')
                jmat[i, j] = 1 / (n1 * n2) ** 2 * (s1 + s2)
        return jmat

    def jmatrix_sample(self, model: CovarianceModel, params_for_gradient: Parameters, n_sims: int = 1000,
                       block_size: int = 100) -> np.ndarray:
        """
        Computes the sample covariance matrix of the gradient of the debiased Whittle likelihood from
        simulated realisations.

        Parameters
        ----------
        model
            Covariance model to sample from
        params_for_gradient
            Parameters with respect to which we take the gradient
        n_sims
            Number of samples used for the estimate covariance matrix
        block_size
            Number of samples per simulations. A higher number should improve
            computational efficiency, but for large grids this may cause
            Out Of Memory issues.

        Returns
        -------
        np.ndarray
            Sample covariance matrix of the gradient of the likelihood
        """
        # TODO here we could simulate independent realizations "in block" as long as we have enough memory
        sampler = SamplerOnRectangularGrid(model, self.expected_periodogram.grid)
        sampler.n_sims = block_size
        gradients = []
        for i_sample in range(n_sims):
            z = sampler()
            _, grad = self(z, model, params_for_gradient)
            gradients.append(grad)
        gradients = np.array(gradients)
        return np.cov(gradients.T)


    def variance_of_estimates(self, model: CovarianceModel, params: Parameters, jmat: np.ndarray = None):
        """
        Compute the covariance matrix of the estimated parameters specified by params under the specified
        covariance model.

        Parameters
        ----------
        model
            Covariance model
        params
            Estimated parameters. The method returns the covariance matrix of estimates of those parameters
        jmat
            The variance of the score, if it has already been pre-computed. If not provided, it is computed
            exactly which can be computationally expensive.

        Returns
        -------
        cov_mat
            Covariance matrix of the parameter estimates.
        """
        hmat = self.fisher(model, params)
        if jmat is None:
            jmat = self.jmatrix(model, params)
        return np.dot(inv(hmat), np.dot(jmat, inv(hmat)))


class Estimator:
    def __init__(self, likelihood: DebiasedWhittle, use_gradients: bool = False, max_iter=100, optim_options=None):
        self.likelihood = likelihood
        self.max_iter = max_iter
        self.use_gradients = use_gradients
        self.optim_options = optim_options

    def __call__(self, model: CovarianceModel, z: Union[np.ndarray, SampleOnRectangularGrid], opt_callback: Callable = None):
        free_params = model.free_params

        # function to be optimized.
        # In the case where the use_gradients property is True, it returns a 2-tuple,
        # the function value and its gradient.
        func = self._get_opt_func(model, free_params, z, self.use_gradients)

        bounds = model.free_param_bounds
        init_guess = numpy.array(free_params.init_guesses)
        x, f, d = fmin_l_bfgs_b(func, init_guess, bounds=bounds, approx_grad=not self.use_gradients,
                      maxiter=self.max_iter, callback=opt_callback, **self.optim_options)
        #minimize(func, init_guess, bounds=bounds, callback=opt_callback)
        return model

    def _get_opt_func(self, model, free_params, z, use_gradients):
        if not use_gradients:
            def func(param_values):
                updates = dict(zip(free_params.names, param_values))
                free_params.update_values(updates)
                return self.likelihood(z, model)
        else:
            def func(param_values):
                updates = dict(zip(free_params.names, param_values))
                free_params.update_values(updates)
                return self.likelihood(z, model, params_for_gradient=free_params)
        return func

    def covmat(self, model: CovarianceModel, params: Parameters = None):
        """Computes an approximation to the covariance matrix of the estimated vector"""
        jmat = self.likelihood.jmatrix(model, params)
        hmat = self.likelihood.fisher(model, params)
        print(f'{jmat=}, {hmat=}')
        return np.dot(inv(hmat), np.dot(jmat, inv(hmat)))