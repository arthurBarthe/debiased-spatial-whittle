from functools import cached_property
import numpy as np
from numpy.fft import fftshift
from scipy.stats import chi2, chisquare, uniform, norm
from matplotlib import pyplot as plt

from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.utils import plot_fourier_values


class GoodnessOfFitSimonsOlhede:
    """
    Class to carry out a goodness-of-fit analysis that relies on spectral residuals as defined in
    Simons & Olhede 2013.
    """

    def __init__(
        self, model: CovarianceModel, grid: RectangularGrid, sample: np.ndarray
    ):
        self.model = model
        self.grid = grid
        self.sample = sample
        self.periodogram_computer = Periodogram()

    def compute_residuals(self):
        r"""
        Compute spectral residuals.

        Notes
        -----
        The residuals are defined according to,

        $$
            X(k) = \frac{I(\mathbf{k})}{\overline{I}(\mathbf{k}; \widehat{\boldsymbol{\theta}})},
        $$

        where $\widehat{\boldsymbol{\theta}}$ is the model's parameter estimates. Under the correct model,
        these residuals are expected to (approximately) follow a chi-square distribution with two degrees of freedom
        multiplied by a factor one half.

        Returns
        -------
        residuals
            Array of spectral residuals.
        """
        periodogram = self.periodogram_computer(self.sample)
        ep = ExpectedPeriodogram(self.grid, self.periodogram_computer)(self.model)
        residuals = periodogram / ep
        return residuals

    def plot(self):
        """
        Generates two plots regarding the distribution of the spectral-domain residuals. A Fourier-space
        plot of the residual values, and a qq-plot against the approximate theoretical distribution.

        Examples
        --------
        >>> from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
        >>> from debiased_spatial_whittle.grids import RectangularGrid
        >>> from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
        >>> model = ExponentialModel(rho=8)
        >>> grid = RectangularGrid((128, 128))
        >>> sample = SamplerOnRectangularGrid(model, grid)()
        >>> gof = GoodnessOfFitSimonsOlhede(model, grid, sample)
        >>> gof.plot()
        >>> model = SquaredExponentialModel(rho=8)
        >>> gof = GoodnessOfFitSimonsOlhede(model, grid, sample)
        >>> gof.plot()
        """
        residuals = self.compute_residuals()
        # spatial plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plot_fourier_values(self.grid, fftshift(residuals), ax=ax, vmin=0, vmax=6)
        ax.set_title("Fourier residuals")
        # qq plot
        dist = chi2(df=2)
        ps = np.linspace(0, 1, 500)
        th_quantiles = dist.ppf(ps) / 2
        emp_quantiles = np.quantile(residuals, ps)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(th_quantiles, emp_quantiles, "-*")
        ax.plot(th_quantiles, th_quantiles)
        ax.set_aspect("equal")
        ax.set_title("QQ-plot of Fourier residuals")
        plt.show()

    def compute_diagnostic_statistic(self):
        residuals = self.compute_residuals()
        statistic = np.mean((residuals - 1) ** 2)
        th_variance = 8 / self.grid.n_points
        statistic = -np.abs(statistic - 1) * np.sqrt(1 / th_variance)
        p_value = norm.cdf(statistic) * 2
        return statistic, p_value


class GoodnessOfFit:
    """
    Class to carry out a goodness-of-fit analysis that relies on spectral residuals expected to be U(0, 1)
    distributed.
    """

    def __init__(
        self,
        model: CovarianceModel,
        grid: RectangularGrid,
        sample: np.ndarray,
        n_bins: int = 10,
    ):
        self.model = model
        self.grid = grid
        self.sample = sample
        self.n_bins = n_bins
        self.periodogram_computer = Periodogram()
        self.bootstrap = True

    @cached_property
    def sampler(self):
        return SamplerOnRectangularGrid(self.model, self.grid)

    def compute_residuals(self, sample, model):
        r"""
        Compute spectral residuals.

        Notes
        -----
        The residuals are defined according to,

        $$
            X(k) = 1 - \exp\left(-\frac{I(\mathbf{k})}{\overline{I}(\mathbf{k}; \widehat{\boldsymbol{\theta}})}\right),
        $$

        where $\widehat{\boldsymbol{\theta}}$ is the model's parameter estimates. Under the correct model, the
        residuals defined above are expected to (approximately) follow a Uniform distribution over the unit interval.

        Returns
        -------
        residuals
            Array of spectral residuals.
        """
        periodogram = self.periodogram_computer(sample)
        ep = ExpectedPeriodogram(self.grid, self.periodogram_computer)(model)
        residuals = 1 - np.exp(-periodogram / ep)
        return residuals

    def compute_diagnostic_statistic(
        self, sample: np.ndarray = None, model: CovarianceModel = None
    ) -> tuple[float, float]:
        """
        Compute a diagnostic statistic from the residuals. The provided p-value is derived under the assumption that Fourier residuals are
        not correlated. This might be a poor approximation in practice, leading to incorrectly small p-values.
        To address this issue, one can use the method p_value instead.

        Parameters
        ----------
        sample
            random field data
        model
            fitted covariance model

        Returns
        -------
        statistic, p-value
            value of the statistic and the corresponding theoretical p-value
        """
        if sample is None:
            sample = self.sample
            model = self.model
        residuals = self.compute_residuals(sample, model).flatten()
        bin_counts = np.bincount((residuals * self.n_bins).astype(np.int64))
        statistic, pvalue = chisquare(bin_counts)
        return statistic, pvalue

    def p_value(self, statistic: float, n_sim: int = 20) -> float:
        """
        Compute a p-value for the diagnostic statistic. The p-value is empirical as it relies on simulations
        from the model.

        Parameters
        ----------
        statistic
            Value of the statistic obtained from the data
        n_sim
            Number of simulated random fields used to compute the empirical p-value.

        Returns
        -------
        p-value
            p-value for the model
        """
        statistic_values = []
        dbw = DebiasedWhittle(
            self.periodogram_computer,
            ExpectedPeriodogram(self.grid, self.periodogram_computer),
        )
        estimator = Estimator(dbw)
        for i in range(n_sim):
            sample = self.sampler()
            if self.bootstrap:
                model_est = self.get_model_est()
                estimator(model_est, sample)
                statistic_value, _ = self.compute_diagnostic_statistic(
                    sample, model_est
                )
            else:
                statistic_value, _ = self.compute_diagnostic_statistic(
                    sample, self.model
                )
            statistic_values.append(statistic_value)
        return np.mean(statistic <= statistic_values)

    def plot(self):
        """
        Generates two plots regarding the distribution of the spectral-domain residuals. A Fourier-space
        plot of the residual values, and a qq-plot against the approximate theoretical distribution.

        Examples
        --------
        >>> from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
        >>> from debiased_spatial_whittle.grids import RectangularGrid
        >>> from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
        >>> model = ExponentialModel(rho=8)
        >>> grid = RectangularGrid((128, 128))
        >>> sample = SamplerOnRectangularGrid(model, grid)()
        >>> gof = GoodnessOfFitSimonsOlhede(model, grid, sample)
        >>> gof.plot()
        >>> model = SquaredExponentialModel(rho=8)
        >>> gof = GoodnessOfFitSimonsOlhede(model, grid, sample)
        >>> gof.plot()
        """
        residuals = self.compute_residuals(self.sample, self.model)
        # spatial plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plot_fourier_values(self.grid, fftshift(residuals), ax=ax, vmin=0, vmax=1)
        ax.set_title("Fourier residuals")
        # qq plot
        dist = uniform()
        ps = np.linspace(0, 1, 500)
        th_quantiles = dist.ppf(ps)
        emp_quantiles = np.quantile(residuals, ps)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(th_quantiles, emp_quantiles, "-*")
        ax.plot(th_quantiles, th_quantiles)
        ax.set_aspect("equal")
        ax.set_title("QQ-plot of Fourier residuals")
        plt.show()


class ModelDiagnostic:
    """
    A generic class to carry out a simple model diagnostic to verify:
    1. that the expected periodogram matches a sample average of periodograms
    2. the distribution of estimates
    """

    # the values below are used as default
    _run_estimation = True
    _n_samples = 100

    def __init__(self, model: CovarianceModel, grid: RectangularGrid):
        self.model = model
        self.grid = grid
        self.sampler = None
        self.periodogram = Periodogram()
        self._samples = []

    @property
    def expected_periodogram(self):
        return ExpectedPeriodogram(self.grid, self.periodogram)(self.model)

    @property
    def likelihood(self):
        return DebiasedWhittle(self.periodogram, self.expected_periodogram)

    @property
    def estimator(self):
        return Estimator(self.likelihood)

    @property
    def run_estimation(self):
        return self._run_estimation

    @run_estimation.setter
    def run_estimation(self, value: bool):
        self._run_estimation = value

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value: int):
        self._n_samples = value

    @property
    def sample_periodograms(self):
        return np.stack([sample["periodogram"] for sample in self._samples])

    @property
    def sample_data(self):
        return np.stack([sample["data"] for sample in self._samples])

    def _run_sample(self):
        z = self.sampler()
        p = self.periodogram(z)
        if self.run_estimation:
            model_est = self.get_estimation_model()
            e = self.estimator(model_est, z)
        self._samples.append(dict(data=z, periodogram=p, estimate=e))

    def run(self):
        # run Monte Carlo simulations
        for i in range(self.n_samples):
            self._run_sample()

    def compare_p_ep(self):
        """
        Compare sample average of periodograms to expected periodogram

        Returns
        -------

        """
        fig = plt.figure()
        ax = fig.add_subplot()


from numpy import ndarray
from scipy import stats


class DiagnosticTest:
    def __init__(self, I: ndarray, f: ndarray, alpha: float = 0.05):
        self._I = I
        self._f = f
        self._n = np.prod(I.shape)  # TODO: missing observations
        self._alpha = alpha

    @property
    def I(self):
        return self._I

    @property
    def n(self):
        return self._n

    @property
    def test_statistic(self):
        return np.mean(self.I / self.f)

    @property
    def residuals(self):
        return self.I / self.f

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, arr):
        self._f = arr
        self()

    def __repr__(self):
        return "Goodness-of-fit spectrum test"

    @staticmethod
    def construct_res(
        pass_test: bool, test_statistic: float, confidence_interval: list
    ):
        return locals()

    def __call__(self, alpha: float = 0.05):
        lp, up = alpha / 2, 1 - alpha / 2
        lb, ub = stats.norm.ppf(
            [lp, up], loc=1.0, scale=np.sqrt(1 / self.n)
        )  # normal approx
        success = lb < self.test_statistic < ub
        CI = [round(lb, 3), round(ub, 3)]

        self.res = self.construct_res(success, round(self.test_statistic, 3), CI)
        for k, v in self.res.items():
            print(f'{f"{k}":>20}:', v)
        return self.res
