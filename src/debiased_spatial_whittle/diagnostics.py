import numpy as np
from matplotlib import pyplot as plt
from debiased_spatial_whittle.models import CovarianceModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator


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
        return np.stack([sample['periodogram'] for sample in self._samples])

    @property
    def sample_data(self):
        return np.stack([sample['data'] for sample in self._samples])

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
        
    def __init__(self, I: ndarray, f: ndarray, alpha:float=0.05):
        self._I = I
        self._f = f
        self._n = np.prod(I.shape)   # TODO: missing observations 
        self._alpha = alpha
        
    @property
    def I(self):
        return self._I
    
    @property
    def n(self):
        return self._n
    
    @property
    def test_statistic(self):
        return np.mean(self.I/self.f)
    
    @property
    def residuals(self):
        return self.I/self.f
    
    @property
    def f(self):
        return self._f
    
    @f.setter
    def f(self, arr):
        self._f = arr
        self()
    
    def __repr__(self):
        return 'Goodness-of-fit spectrum test'
    
    @staticmethod
    def construct_res(pass_test:bool, test_statistic:float, confidence_interval:list):
        return locals()

    def __call__(self, alpha:float=0.05):
        
        lp,up = alpha/2, 1-alpha/2
        lb, ub = stats.norm.ppf([lp,up], loc=1., scale=np.sqrt(1/self.n))  # normal approx
        success = lb<self.test_statistic<ub
        CI = [round(lb,3), round(ub,3)]

        self.res = self.construct_res(success, round(self.test_statistic,3), CI)       
        for k, v in self.res.items():
            print (f'{f"{k}":>20}:', v)
        return self.res
