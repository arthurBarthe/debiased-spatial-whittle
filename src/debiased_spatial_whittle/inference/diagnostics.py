from functools import cached_property
import numpy as np
from scipy.stats import chisquare
from matplotlib import pyplot as plt
from debiased_spatial_whittle.models.base import CovarianceModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid


class GoodnessOfFit:
    """
    Class to perform a goodness of fit analysis between a model and a sampled random field.
    """
    def __init__(
        self, model: CovarianceModel, grid: RectangularGrid, sample, n_bins: int = 10
    ):
        """
        Parameters
        ----------
        model
            Covariance model for the data
        grid
            Sampling grid
        sample
            sampled random field data
        n_bins
            Number of bins used in the goodness-of-fit analysis
        """
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
        periodogram = self.periodogram_computer(sample)
        ep = ExpectedPeriodogram(self.grid, self.periodogram_computer)(model)
        residuals = 1 - np.exp(-periodogram / ep)
        return residuals

    def compute_diagnostic_statistic(self, sample=None, model=None):
        if sample is None:
            sample = self.sample
            model = self.model
        residuals = self.compute_residuals(sample, model).flatten()
        bin_counts = np.bincount((residuals * self.n_bins).astype(np.int64))
        statistic, pvalue = chisquare(bin_counts)
        return statistic, pvalue

    def p_value(self, statistic: float, n_sim: int = 20):
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