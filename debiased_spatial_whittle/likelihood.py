import warnings
import numpy as np
from numpy.fft import fftn
from scipy.optimize import fmin_l_bfgs_b
from scipy.signal import hanning
from .expected_periodogram import compute_ep


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
    return 1 / n * np.sum((e_per - per) * e_per_prime / e_per ** 2)


def fit(y, grid, cov_func, init_guess, fold=True, cov_func_prime=None, taper=False):
    # apply taper if required
    if taper:
        y, taper = apply_taper(y)
        grid = grid * taper

    per = periodogram(y, grid, fold=fold)

    if cov_func_prime is not None:
        def opt_func(x):
            e_per = compute_ep(lambda lags: cov_func(lags, *x), grid, fold=fold)
            e_per_prime = compute_ep(lambda lags: (cov_func_prime(lags, *x))[0], grid, fold=fold)
            return whittle(per, e_per), whittle_prime(per, e_per, e_per_prime)

        est_, fval, info = fmin_l_bfgs_b(lambda x: opt_func(x),
                                         init_guess,
                                         maxiter=100,
                                         bounds=[(0.01, 1000), ] * len(init_guess))
    else:
        def opt_func(x):
            e_per = compute_ep(lambda lags: cov_func(lags, *x), grid, fold=fold)
            return whittle(per, e_per)
        est_, fval, info = fmin_l_bfgs_b(lambda x: opt_func(x),
                                         init_guess,
                                         maxiter=100,
                                         approx_grad=True,
                                         bounds=[(0.01, 1000), ] * len(init_guess))

    if info['warnflag'] != 0:
        warnings.warn('Issue during optimization')

    return est_



#########NEW OOP version
from expected_periodogram import ExpectedPeriodogram
from models import CovarianceModel

class Periodogram:
    """Class that allows to define a periodogram"""

    def __init__(self, taper = None, scaling='ortho'):
        if taper is None:
            self.taper = 1
        self.scaling = scaling
        #TODO add possibility to not fold?
        self.fold = True

    def __call__(self, z: np.ndarray):
        z_taper = z * self.taper
        f = np.abs(fftn(z, norm=self.scaling))**2
        return f



class DebiasedWhittle:
    def __init__(self, periodogram: Periodogram, expected_periodogram: ExpectedPeriodogram):
        self.periodogram = periodogram
        self.expected_periodogram = expected_periodogram

    def get_h(self, model):
        pass

    def get_j(self, model):
        pass

    def __call__(self, z: np.ndarray, model: CovarianceModel):
        # TODO add a class sample which contains the data and the grid?
        """Computes the likelihood for this data"""
        p = self.periodogram(z)
        ep = self.expected_periodogram(model)
        return 1 / z.shape[0] / z.shape[1] * np.sum(np.log(ep) + p / ep)


class Estimator:
    def __init__(self, likelihood, max_iter=100):
        self.likelihood = likelihood
        self.max_iter = 100

    def __call__(self, model: CovarianceModel, z: np.ndarray):
        free_params = model.free_params
        free_params_names = free_params.names

        # function to be optimized
        def func(param_values):
            updates = dict(zip(free_params_names, param_values))
            free_params.update_values(updates)
            return self.likelihood(z, model)

        bounds = model.free_param_bounds
        init_guess = np.array(free_params.init_guesses)
        # TODO add the case where gradient is available
        fmin_l_bfgs_b(func, init_guess, bounds=bounds, approx_grad=True, maxiter=self.max_iter)
        return model
