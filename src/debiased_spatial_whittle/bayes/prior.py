from numpy import ndarray
import autograd.numpy as np
from autograd import grad

from scipy import stats
from autograd.scipy import stats as ag_stats

from abc import ABC, abstractmethod
from typing import Union

class Prior(ABC):
    # TODO: do on regular parameter space!! add inv_transform to prior and likelihood class!!
    
    @property
    @abstractmethod
    def n_params(self):
        pass
    
    @abstractmethod
    def __call__(self):
        '''logpdf (autograd)'''
        pass

    @abstractmethod
    def pdf(self):
        pass

    @abstractmethod
    def sim(self):
        pass

class GaussianPrior(Prior):
    # TODO: non-gaussian priors
    def __init__(self, mean: ndarray, cov: ndarray):
        self._mean = mean
        self._cov = cov
        assert len(mean) == len(cov), 'dimensions must match'
        
        self._dist = stats.multivariate_normal(mean=self.mean, cov=self.cov)  # scipy version, has more methods
    
    @property
    def n_params(self):
        return len(self.mean)
    
    @property    
    def mean(self):
        return self._mean
    
    @property
    def cov(self):
        return self._cov
    
    def __repr__(self):
        return f'''Gaussian prior on the unrestricted space with   \
                 \n Mean: {self.mean.round(3)},                    \
                 \n Cov: \n {self.cov.round(3)}.'''
    
    def __call__(self, x: ndarray) -> float:
        '''logpdf (autograd)'''
        return ag_stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)
    
    def pdf(self, x: ndarray) -> float:
        '''pdf (autograd)'''
        return ag_stats.multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)
        
    def sim(self, size: Union[int,ndarray]=1) -> ndarray:
        return self._dist.rvs(size)




from numpy.testing import assert_allclose

def test_logpdf():
    d=5
    mean = np.zeros(d)
    cov = np.eye(d)

    x = np.zeros(d)
    prior_val_true = stats.multivariate_normal(mean, cov).logpdf(x)
    prior_val_test = GaussianPrior(mean, cov)(x)
    assert_allclose(prior_val_true , prior_val_test)
    

def test_pdf():
    d=10
    mean = np.zeros(d)
    cov = np.eye(d)

    x = np.zeros(d)
    prior_val_true = stats.multivariate_normal(mean, cov).pdf(x)
    prior_val_test = GaussianPrior(mean, cov).pdf(x)
    assert_allclose(prior_val_true , prior_val_test)
    
def test_grad_logpdf():
    # from autograd import grad
    d=20
    mean = np.zeros(d)
    cov = np.eye(d)

    x = np.zeros(d)
    prior_grad_true = np.zeros(d)
    prior_grad_test = grad(GaussianPrior(mean, cov))(x)
    assert_allclose(prior_grad_true , prior_grad_test)
    
if __name__ == "__main__":
    test_logpdf()
    test_pdf()
    test_grad_logpdf()
