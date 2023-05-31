# This example implements the real-data example from the paper. The data used are remote-sensing measurements of Venus'
# topography.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import MaternCovarianceModel, MaternModel
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator

# load the topography data and standardize by the std
z = scipy.io.loadmat('Frederik53.mat')['topodata']
z = (z - np.mean(z)) / np.std(z)

grid = RectangularGrid(z.shape)
periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
estimator = Estimator(debiased_whittle)

# frequency mask corresponding to the data processing
from numpy.fft import fftfreq
m, n = z.shape
x, y = np.meshgrid(fftfreq(m) * 2 * np.pi, fftfreq(n) * 2 * np.pi, indexing='ij')
freq_norm = np.sqrt(x ** 2 + y ** 2)
frequency_mask = freq_norm < np.pi
print(z.shape, frequency_mask.shape)

plt.figure()
plt.imshow(frequency_mask)
plt.show()

debiased_whittle.frequency_mask = frequency_mask

# plot
plt.figure()
plt.imshow(z, origin='lower', cmap='bwr')
plt.show()

# periodogram plot
from numpy.fft import fftshift
plt.figure()
per = periodogram(z)
plt.imshow(fftshift(10 * np.log10(per) * frequency_mask))
plt.show()

model = MaternModel()
model.rho.init_guess=10
model.sigma.init_guess = 1
model.nu.init_guess = 1

print(model)
print(estimator(model, z))

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
sampler = SamplerOnRectangularGrid(model, grid)
z_sim = sampler()
plt.figure()
plt.imshow(z_sim, cmap='bwr')
plt.show()