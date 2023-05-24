import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian
# from debiased_spatial_whittle.bayes_old import DeWhittle2


fftn = np.fft.fftn

np.random.seed(1252147)

n = (64, 64)

mask = np.ones(n)
n_missing = 10
missing_idxs = np.random.randint(n[0], size=(n_missing,2))
mask[tuple(missing_idxs.T)] = 0.

plt.imshow(mask, cmap='Greys', origin='lower')
plt.show()

grid = RectangularGrid(n, mask=mask)

model = SquaredExponentialModel()
model.rho = 10
model.sigma = 1
model.nugget=0.1
sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

plt.imshow(z, origin='lower')
plt.show()

params = np.array([10.,1.])

dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1)
dw.fit(None, prior=False)
MLEs = dw.sim_MLEs(params, niter=10)
plot_marginals([MLEs], np.log(params))


missing_point = missing_idxs[0]
lags = np.meshgrid(*(np.arange(0, _n) for _n in n), indexing='ij') # TODO: deltas

d = np.sqrt(sum(((lag-lag[(*missing_point,)])**2 for i,lag in enumerate(lags))))

ind = np.unravel_index(np.argsort(d, axis=None), d.shape)
model
