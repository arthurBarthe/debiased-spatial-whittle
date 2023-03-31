import sys

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

params = np.array([10.,1.5])
print(dw.loglik(params))
print(dw.logpost(params))

