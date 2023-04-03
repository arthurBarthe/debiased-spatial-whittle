import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv
from debiased_spatial_whittle.bayes import DeWhittle
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals


fftn = np.fft.fftn

np.random.seed(1252149)

n = (128, 128)
rho, sigma = 10, 1

grid = RectangularGrid(n)
model = SquaredExponentialModel()
model.rho = rho
model.sigma = sigma

per = Periodogram()
ep = ExpectedPeriodogram(grid, per)
db = DebiasedWhittle(per, ep)

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

I = per(z)
eI = fftshift(ep(model))
# plt.imshow(fftshift(I))
# plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(z, origin='lower', cmap='Spectral')
plt.show()


dw = DeWhittle(z, grid, SquaredExponentialModel())
eI = dw.expected_periodogram([10,1])
# plt.imshow(fftshift(eI))
# plt.show()

params = np.log([10.,1.])
print(dw.loglik(params))
print(dw.logpost(params))

from autograd import grad, hessian
ll = lambda x: dw.loglik(x)
print(grad(ll)(params))

dw.fit(None, prior=True)
# stop

MLEs1 = dw.estimate_standard_errors(params, monte_carlo=True, niter=10000,  const='whittle')
MLEs2 = dw.estimate_standard_errors(params, monte_carlo=True, niter=10000,  const='dewhittle')
print(np.cov(MLEs1.T).round(3))
print(np.cov(MLEs2.T).round(3))


title = 'MLE distribution'
legend_labels = ['Whittle constant', 'deWhittle constant']
plot_marginals([MLEs1, MLEs2], params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))


post_samps,A = dw.RW_MH(5000)
title = 'posterior'
legend_labels = ['deWhittle posterior']
plot_marginals([post_samps], params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))




