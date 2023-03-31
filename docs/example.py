import numpy as np
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

fftn = np.fft.fftn

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
eI = dw.expected_periodogram([10,2])
# plt.imshow(fftshift(eI))
# plt.show()

params = np.log([10.,1.])
print(dw.loglik(params))
print(dw.logpost(params))

from autograd import grad, hessian
ll = lambda x: dw.loglik(x)
print(grad(ll)(params))

dw.fit(None, prior=True)
