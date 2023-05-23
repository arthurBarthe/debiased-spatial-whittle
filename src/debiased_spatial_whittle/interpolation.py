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

grid = RectangularGrid(n)

model = SquaredExponentialModel()
model.rho = 10
model.sigma = 1
model.nugget=0.1
sampler = SamplerOnRectangularGrid(model, grid)
z_ = sampler()
z = z_ * mask

plt.imshow(z, origin='lower')
plt.show()

params = np.array([10.,1.])

dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1)
dw.fit(None, prior=False)
MLEs = dw.sim_MLEs(params, niter=10)
plot_marginals([MLEs], np.log(params))


missing_point = missing_idxs[0]

lags = grid.lag_matrix
covMat = model(lags)
# lags = np.meshgrid(*(np.arange(0, _n) for _n in n), indexing='ij') # TODO: deltas

xs = np.meshgrid(*(np.arange(0, m) for m in n), indexing='ij')
X  = np.array(xs).reshape(2,np.prod(n)).T

d2 = np.sum((X - missing_point)**2, axis=1)
nugget_effect = model.nugget.value*np.all(d2 == 0, axis=0)
acf = model.sigma.value ** 2 * np.exp(- 0.5*d2 / model.rho.value ** 2) + nugget_effect


# acf @ inv_covMat @ z.flatten()
# z.flatten() @ inv_covMat @ acf

# np.dot(z[mask], weights[mask.flatten()])
# var = 1.1 - acf@ inv_covMat @ acf

inv_covMat = np.linalg.inv(covMat)
weights  = inv_covMat @ acf

plt.imshow(covMat)
plt.show()

plt.plot(acf)
plt.show()
stop

ndim = len(n)

lags_ = lags - missing_point.reshape(-1, *[1]*ndim)

acf = model(lags_)
plt.imshow(acf, origin='lower')
plt.show()

stop

d = np.sqrt(sum(((lag-lag[(*missing_point,)])**2 for i,lag in enumerate(lags))))

ind = np.unravel_index(np.argsort(d, axis=None), d.shape)
model
