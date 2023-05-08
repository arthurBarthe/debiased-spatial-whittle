import numpy as np
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