import numpy as np

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, Parameters
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle
from debiased_spatial_whittle.confidence import McmcDiags, CovarianceFFT


def test_jmat():
    g = RectangularGrid((16, 16))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 10
    jmat = d.jmatrix(model, model.params)
    print(jmat)
    assert np.all(np.diag(jmat) >= 0)


def test_mcmc_jmat():
    g = RectangularGrid((16, 16))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    model = ExponentialModel()
    model.sigma = 1
    model.rho = 5
    f = np.ones((16, 16))
    mcmc = McmcDiags(model, ep, f, f)
    cov_fft = CovarianceFFT(g)
    s1 = cov_fft.exact_summation1(model, ep, f, f, normalize=True)
    mcmc.run(10000)
    print(mcmc.partition_function(), mcmc.estimate())
    print(s1)
    import matplotlib.pyplot as plt
    plt.plot(mcmc.partition_function_trace())
    plt.show()
