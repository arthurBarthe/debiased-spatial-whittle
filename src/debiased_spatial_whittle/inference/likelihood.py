from typing import Callable, Union
from scipy.optimize import minimize, fmin_l_bfgs_b
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.sampling.simulation import SamplerBUCOnRectangularGrid
from debiased_spatial_whittle.models.base import CovarianceModel, ModelParameter
from debiased_spatial_whittle.inference.multivariate_periodogram import (
    Periodogram as MultPeriodogram,
)
from debiased_spatial_whittle.sampling.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.inference.confidence import CovarianceFFT
from debiased_spatial_whittle.backend import BackendManager


xp = BackendManager.get_backend()
slogdet = BackendManager.get_slogdet()
inv = BackendManager.get_inv()
fftn = xp.fft.fftn
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


def whittle_prime(per, e_per, e_per_prime):
    n = prod_list(per.shape)
    if e_per.ndim != e_per_prime.ndim:
        out = []
        for i in range(e_per_prime.shape[-1]):
            out.append(whittle_prime(per, e_per, e_per_prime[..., i]))
        return xp.stack(out, axis=-1)
    e_per_prime = xp.reshape(e_per_prime, per.shape)
    return 1 / n * xp.sum((e_per - per) * e_per_prime / e_per ** 2)


class MultivariateDebiasedWhittle:
    """
    Implements the Debiased Whittle Likelihood for multivariate data. This requires
    the use of a multivariate periodogram.
    Currently, only implemented for bi-variate.

    Attributes
    ----------
    periodogram: MultPeriodogram
        Multivariate periodogram applied to the multivariate random field

    expected_periodogram: ExpectedPeriodogram
        Object used to compute the expectation of the periodogram
    """

    def __init__(
        self, periodogram: MultPeriodogram, expected_periodogram: ExpectedPeriodogram
    ):
        self.periodogram = periodogram
        self.expected_periodogram = expected_periodogram

    def __call__(
        self,
        z: xp.ndarray,
        model: CovarianceModel,
        params_for_gradient: list[ModelParameter] = None,
    ):
        """Computes the likelihood for these data"""
        p = self.periodogram([z[..., 0], z[..., 1]])
        ep = self.expected_periodogram(model)
        n_spatial_dim = p.ndim - 2
        if p.ndim == ep.ndim - 1:
            # multiple model parameter vectors
            p = xp.expand_dims(p, -3)
        ep_inv = inv(ep)
        term1 = slogdet(ep)[1]
        ratio = xp.matmul(ep_inv, p)
        if BackendManager.backend_name in ("numpy", "cupy"):
            term2 = xp.trace(ratio, axis1=-2, axis2=-1)
        elif BackendManager.backend_name == "torch":
            term2 = xp.sum(xp.diagonal(ratio, dim1=-1, dim2=-2), -1)
        whittle = xp.mean(term1 + term2, tuple(range(n_spatial_dim)))
        whittle = xp.real(whittle)
        if BackendManager.backend_name == "torch":
            whittle = whittle.item()
        if not params_for_gradient:
            return whittle
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        ep_inv = xp.expand_dims(ep_inv, -3)
        p = xp.expand_dims(p, -3)
        # the derivative of the log determinant
        d_log_det = xp.trace(xp.matmul(ep_inv, d_ep), axis1=-2, axis2=-1)
        # the derivative the second term
        d_ep_inv = -xp.matmul(ep_inv, xp.matmul(d_ep, ep_inv))
        d_quad_term = xp.trace(xp.matmul(d_ep_inv, p), axis1=-2, axis2=-1)
        # derivative
        d_whittle = xp.mean(d_log_det + d_quad_term, axis=(0, 1))
        return whittle, d_whittle

    def fisher(self, model: CovarianceModel, params_for_gradient: list[ModelParameter]):
        """Provides the expectation of the hessian matrix"""
        n_params = len(params_for_gradient)
        ep = self.expected_periodogram(model)
        ep_inv = inv(ep)
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        h = zeros((n_params, n_params))
        for i1 in range(n_params):
            for i2 in range(n_params):
                d_ep1 = d_ep[..., i1]
                d_ep2 = d_ep[..., i2]
                h[i1, i2] = xp.mean(
                    xp.trace(
                        xp.matmul(ep_inv, xp.matmul(d_ep1, xp.matmul(ep_inv, d_ep2))),
                        axis1=-2,
                        axis2=-1,
                    )
                )
        return h

    def jmatrix_sample(
        self,
        model: CovarianceModel,
        params_for_gradient: list[ModelParameter],
        n_sims: int = 400,
        block_size: int = 100,
    ) -> xp.ndarray:
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
        sampler = SamplerBUCOnRectangularGrid(model, self.expected_periodogram.grid)
        sampler.n_sims = block_size
        gradients = []
        for i_sample in range(n_sims):
            z = sampler()
            _, grad = self(z, model, params_for_gradient)
            gradients.append(grad)
        gradients = xp.array(gradients)
        # enforce real values
        return xp.real(xp.cov(gradients.T))


class DebiasedWhittle:
    """
    Implements the Debiased Whittle likelihood for univariate data.

    Attributes
    ----------
    periodogram: Periodogram
        Periodogram applied to the data

    expected_periodogram: ExpectedPeriodogram
        Object used to compute the expectation of the periodogram

    frequency_mask: ndarray
        mask of zero and ones to select frequencies over which the summation is carried out in the computation of
        the Whittle.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1712)
    >>> from debiased_spatial_whittle.grids.base import RectangularGrid
    >>> from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
    >>> from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
    >>> from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
    >>> grid = RectangularGrid(shape=(256, 256))
    >>> model1 = SquaredExponentialModel()
    >>> model1.rho = 12
    >>> model1.sigma = 1
    >>> model2 = SquaredExponentialModel()
    >>> model2.rho = 4
    >>> model2.sigma = 1
    >>> sampler = SamplerOnRectangularGrid(model1, grid)
    >>> per = Periodogram()
    >>> ep = ExpectedPeriodogram(grid, per)
    >>> dbw = DebiasedWhittle(per, ep)
    >>> sample = sampler()
    >>> dbw(sample, model1), dbw(sample, model2)
    (-8.37515633567113, -7.315812857735173)
    """

    def __init__(
        self, periodogram: Periodogram, expected_periodogram: ExpectedPeriodogram
    ):
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
    def frequency_mask(self, value: xp.ndarray):
        """
        Define a mask in the spectral domain to fit only certain frequencies

        Parameters
        ----------
        value
            mask of zeros and ones
        """
        if value is not None:
            assert (
                value.shape == self.expected_periodogram.grid.n
            ), "shape mismatch between mask and grid"
        self._frequency_mask = value

    def whittle(self, periodogram: xp.ndarray, expected_periodogram: xp.ndarray):
        """
        Compute the Whittle distance between periodogram values and expectation.

        Parameters
        ----------
        periodogram
            periodogram of the data on Fourier grid

        expected_periodogram
            expected periodogram or spectral density values on same Fourier grid

        Returns
        -------
        whittle_value: float
            whittle distance between periodogram and expected periodogram

        Notes
        -----
        In standard use cases, this method should not be called directly. Instead, one should use the __call__
        method.
        """
        if periodogram.ndim == expected_periodogram.ndim - 1:
            # this handles the case of several expected periodograms indexed by the last dimension
            ndim = expected_periodogram.ndim
            periodogram = xp.expand_dims(periodogram, -1)
            frequency_mask = xp.expand_dims(self.frequency_mask, -1)
            return xp.mean(
                (xp.log(expected_periodogram) + periodogram / expected_periodogram)
                * frequency_mask,
                tuple(range(ndim - 1)),
            )
        return xp.mean(
            (xp.log(expected_periodogram) + periodogram / expected_periodogram)
            * self.frequency_mask
        )

    def __call__(
        self,
        sample: xp.ndarray,
        model: CovarianceModel,
        params_for_gradient: list[ModelParameter] = None,
    ) -> xp.float64:
        """
        Computes the Debiased Whittle likelihood for these data

        Parameters
        ----------
        sample: ndarray | SampleOnRectangularGrid
            sample data

        model: CovarianceModel
            covariance model used to compute the likelihood of the data

        params_for_gradient: Parameters, optional
            parameters with respect to which we require the derivative of the likelihood. Default, None

        Returns
        -------
        likelihood: float
            likelihood value

        gradient: ndarray
            gradient with respect to the parameters provided in params_for_gradient. If the latter is None,
            this second output is not returned.
        """
        p = self.periodogram(sample)
        ep = self.expected_periodogram(model)
        whittle = self.whittle(p, ep)
        if not params_for_gradient:
            return whittle
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        d_whittle = whittle_prime(p, ep, d_ep)
        if BackendManager.backend_name == "torch":
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
        expected: float
            Expectation of the Debiased Whittle likelihood under true_model, evaluated at
            eval_model
        """
        ep_true = self.expected_periodogram(true_model)
        ep_eval = self.expected_periodogram(eval_model)
        return xp.sum(xp.log(ep_eval) + ep_true / ep_eval)

    def fisher(self, model: CovarianceModel, params_for_gradient: list[ModelParameter]):
        """
        Provides the Fisher Information Matrix.

        Parameters
        ----------
        model: CovarianceModel
            True covariance model

        params_for_gradient
            Parameters with respect to which the Fisher is obtained

        Returns
        -------
        fisher: ndarray
            Fisher covariance matrix

        Examples
        --------
        >>> from debiased_spatial_whittle.grids.base import RectangularGrid
        >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
        >>> model = ExponentialModel(rho=30, sigma=1.41)
        >>> periodogram = Periodogram()
        >>> grid = RectangularGrid((67, 192))
        >>> ep = ExpectedPeriodogram(grid, periodogram)
        >>> dbw = DebiasedWhittle(periodogram, ep)
        >>> dbw.fisher(model, [model.param.rho, model.param.sigma])
        array([[ 1.03736229e-03, -4.49238561e-02],
               [-4.49238561e-02,  2.01197123e+00]])
        """
        n_params = len(params_for_gradient)
        ep = self.expected_periodogram(model)
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        h = zeros((n_params, n_params))
        for i1 in range(n_params):
            for i2 in range(n_params):
                d_ep1 = d_ep[..., i1]
                d_ep2 = d_ep[..., i2]
                h[i1, i2] = xp.sum(d_ep1 * d_ep2 / ep ** 2)
        return h / self.expected_periodogram.grid.n_points

    def jmatrix(
        self,
        model: CovarianceModel,
        params_for_gradient: list[ModelParameter],
        mcmc_mode: bool = False,
    ):
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
        jmat = xp.zeros((n_params, n_params))
        grid = self.expected_periodogram.grid
        n1, n2 = grid.n
        covariance_fft = CovarianceFFT(grid)
        d_ep = self.expected_periodogram.gradient(model, params_for_gradient)
        ep = self.expected_periodogram(model)

        for i in range(n_params):
            for j in range(n_params):
                # TODO get rid of repeated computations for efficiency
                d_epi = xp.take(d_ep, i, -1)
                d_epj = xp.take(d_ep, j, -1)
                if not mcmc_mode:
                    s1 = covariance_fft.exact_summation1(
                        model, self.expected_periodogram, d_epi / ep**2, d_epj / ep**2
                    )
                else:
                    mcmc = McmcDiags(
                        model, self.expected_periodogram, d_epi / ep, d_epj / ep
                    )
                    mcmc.run(500)
                    s1 = mcmc.estimate()
                # s2 = covariance_fft.exact_summation2(model, self.expected_periodogram, d_epi/ expected_periodogram**2, d_epj / expected_periodogram**2)
                s2 = s1
                print(f"{s1=}, {s2=}")
                jmat[i, j] = 1 / (n1 * n2) ** 2 * (s1 + s2)
        return jmat

    def jmatrix_sample(
        self,
        model: CovarianceModel,
        params_for_gradient: list[ModelParameter],
        n_sims: int = 1000,
        block_size: int = 100,
    ) -> xp.ndarray:
        """
        Computes the sample covariance matrix of the gradient of the debiased Whittle likelihood from
        simulated realisations. Specifically, this simulates n_sims samples from model, computes
        the gradient for each sample using the __call__ method, and computes the sample covariance of those
        gradients.

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

        Examples
        --------
        >>> import numpy.random as nrrandom
        >>> nrrandom.seed(1712)
        >>> from debiased_spatial_whittle.grids.base import RectangularGrid
        >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
        >>> model = ExponentialModel(rho=12, sigma=1.41)
        >>> periodogram = Periodogram()
        >>> grid = RectangularGrid((67, 192))
        >>> ep = ExpectedPeriodogram(grid, periodogram)
        >>> dbw = DebiasedWhittle(periodogram, ep)
        >>> dbw.jmatrix_sample(model, [model.param.rho, model.param.sigma], n_sims=20)
        array([[ 1.79844275e-06, -3.36165062e-05],
               [-3.36165062e-05,  8.20809861e-04]])
        """
        sampler = SamplerOnRectangularGrid(model, self.expected_periodogram.grid)
        sampler.n_sims = block_size
        gradients = []
        for i_sample in range(n_sims):
            z = sampler()
            _, grad = self(z, model, params_for_gradient)
            gradients.append(grad)
        gradients = xp.array(gradients)
        # enforce real values
        return xp.real(xp.cov(gradients.T))

    def variance_of_estimates(
        self,
        model: CovarianceModel,
        params: list[ModelParameter],
        jmat: xp.ndarray = None,
    ):
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

        Examples
        --------
        >>> import numpy.random as nrrandom
        >>> nrrandom.seed(1712)
        >>> from debiased_spatial_whittle.grids.base import RectangularGrid
        >>> from debiased_spatial_whittle.models.univariate import ExponentialModel
        >>> model = ExponentialModel(rho=12., sigma=4.)
        >>> periodogram = Periodogram()
        >>> grid = RectangularGrid((67, 192))
        >>> ep = ExpectedPeriodogram(grid, periodogram)
        >>> dbw = DebiasedWhittle(periodogram, ep)
        >>> jmat = dbw.jmatrix_sample(model, [model.param.rho, model.param.sigma], n_sims=20)
        >>> dbw.variance_of_estimates(model, [model.param.rho, model.param.sigma], jmat)
        array([[8.27761908, 1.34780351],
               [1.34780351, 0.22064392]])
        """
        hmat = self.fisher(model, params)
        if jmat is None:
            jmat = self.jmatrix(model, params)
        return xp.dot(inv(hmat), xp.dot(jmat, inv(hmat)))


class Estimator:
    """
    Class to define an estimator that uses a likelihood.

    Attributes
    ----------
    likelihood: DebiasedWhittle
        Debiased Whittle likelihood used for fitting.

    use_gradients: bool
        Whether to use gradients in the optimization procedure. Not working at the moment.

    max_iter: int
        Maximum number of iterations of the optimization procedure

    optim_options: dict
        Additional options passed to the optimizer.

    method: string
        Optimization procedure. Should be one of the methods available in scipy's local or global optimizers.
    """

    def __init__(
        self,
        likelihood: DebiasedWhittle,
        use_gradients: bool = False,
        max_iter: int = 100,
        optim_options: dict = dict(),
        method: str = "L-BFGS-B",
    ):
        """

        Parameters
        ----------
        likelihood
            Debiased Whittle likelihood used for fitting.

        use_gradients
            Whether to use gradients in the optimization procedure

        max_iter
            Maximum number of iterations of the optimization procedure

        optim_options
            Additional options passed to the optimizer.

        method
            Optimization procedure
        """
        self.likelihood = likelihood
        self.max_iter = max_iter
        self.use_gradients = use_gradients
        self.optim_options = optim_options
        self.method = method
        self.f_opt = None
        self.f_info = None

    @property
    def use_gradients(self):
        return self._use_gradients

    @use_gradients.setter
    def use_gradients(self, value: bool):
        if value:
            raise NotImplementedError(
                "The use of gradients for optimization is currently not implemented."
            )
        else:
            self._use_gradients = value

    def __call__(
        self,
        model: CovarianceModel,
        sample: Union[xp.ndarray, SampleOnRectangularGrid],
        opt_callback: Callable = None,
    ):
        """
        Fits the passed covariance model to the passed data.

        Parameters
        ----------
        model: CovarianceModel
            Covariance model to be fitted to the data. Only free parameters are estimated, that is parameters
            of the covariance model set to None.

        sample: ndarray | SampleOnRectangularGrid
            Sampled random field

        opt_callback: function handle
            Callback function called by the optimizer

        Returns
        -------
        model: CovarianceModel
            The fitted covariance model

        Notes
        -----
        This directly updates the parameters of the passed covariance model.
        """
        free_params = model.free_parameters

        # function to be optimized.
        # In the case where the use_gradients property is True, it returns a 2-tuple,
        # the function value and its gradient.
        func = self._get_opt_func(model, sample, self.use_gradients)
        if self.use_gradients:
            # TODO: inefficient, we call func twice
            opt_func = lambda x: func(x)[0]
            jac = lambda x: func(x)[1]
        else:
            opt_func = func

        bounds = model.free_parameter_bounds_to_list_deep()
        # np.to_cpu ensures conversion to numpy array, necessary for the optimizer
        x0 = xp.to_cpu(model.free_parameter_values_to_array_deep())

        if self.method in (
            "shgo",
            "direct",
            "differential_evolution",
            "dual_annealing",
        ):
            import scipy

            try:
                opt_result = getattr(scipy.optimize, self.method)(
                    opt_func,
                    bounds=bounds,
                    callback=opt_callback,
                    x0=x0,
                    **self.optim_options,
                )
            except TypeError as e:
                print(e)
                print("Trying again without passing x0...")
                opt_result = getattr(scipy.optimize, self.method)(
                    opt_func, bounds=bounds, callback=opt_callback, **self.optim_options
                )
        else:
            if self.use_gradients:
                opt_result = minimize(
                    opt_func,
                    x0,
                    jac=jac,
                    method=self.method,
                    bounds=bounds,
                    callback=opt_callback,
                    options=self.optim_options,
                )
            else:
                opt_result = minimize(
                    opt_func,
                    x0,
                    method=self.method,
                    bounds=bounds,
                    callback=opt_callback,
                    options=self.optim_options,
                )
        model.update_free_parameters(opt_result.x)
        self.opt_result = opt_result
        return model

    def _get_opt_func(self, model: CovarianceModel, z, use_gradients):
        if not use_gradients:

            def func(param_values):
                model.update_free_parameters(param_values)
                return self.likelihood(z, model).item()
        else:
            # TODO not updated for new models
            def func(param_values):
                model.update_free_parameters(param_values)
                free_params = model.get_free_parameters_deep()
                lkh, grad = self.likelihood(z, model, params_for_gradient=free_params)
                return lkh.item(), grad

        return func

    def covmat(self, model: CovarianceModel, params: list[ModelParameter] = None):
        """
        Compute an approximate covariance matrix of the parameter estimates under the specified covariance model.

        Parameters
        ----------
        model
            True covariance model

        params
            estimated parameters

        Returns
        -------
        covmat: ndarray
            Covariance matrix.
        """
        jmat = self.likelihood.jmatrix(model, params)
        hmat = self.likelihood.fisher(model, params)
        return xp.dot(inv(hmat), xp.dot(jmat, inv(hmat)))
