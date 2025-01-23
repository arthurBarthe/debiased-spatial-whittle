import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    MaternModel,
)
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import (
    Periodogram,
    ExpectedPeriodogram,
    compute_ep,
)
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian
# from debiased_spatial_whittle.bayes_old import DeWhittle2


fftn = np.fft.fftn

np.random.seed(1252147)

n = (64, 64)

mask = np.ones(n)

n_missing = 10
missing_idxs = np.random.randint(n[0], size=(n_missing, 2))
mask[tuple(missing_idxs.T)] = 0.0
m = mask.astype(bool)

plt.imshow(mask, cmap="Greys", origin="lower")
plt.show()

grid = RectangularGrid(n)

model = SquaredExponentialModel()
model.rho = 10
model.sigma = 1
model.nugget = 0.1
sampler = SamplerOnRectangularGrid(model, grid)
z_ = sampler()
z = z_ * mask

plt.imshow(z, origin="lower")
plt.show()

params = np.array([10.0, 1.0])

grid = RectangularGrid(n, mask=m)
dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1)
dw.fit(None, prior=False)
MLEs = dw.sim_MLEs(params, niter=10)
plot_marginals([MLEs], np.log(params))

stop

missing_idxs

a = np.where(m.flatten())

# masks = [np.ones(s, dtype=bool) for s in n]
xs = [np.arange(s, dtype=np.int64) for s in n]
grid = np.meshgrid(*xs, indexing="ij")
grid_vec = [g.reshape((-1, 1))[a] for g in grid]
lags = [g - g.T for g in grid_vec]
# return np.array(lags)

d2_1 = sum(lag**2 for lag in lags)


def get_lags(mask: np.ndarray):
    # mask = mask.astype(bool)
    # flat_idxs = np.where(mask.flatten())
    # xs = [np.arange(s, dtype=np.int64) for s in n]
    # grid = np.meshgrid(*xs, indexing='ij')
    # grid_vec = [g.reshape((-1, 1))[flat_idxs] for g in grid]
    grid_vec = np.argwhere(mask)[None].T
    lags = grid_vec - np.transpose(
        grid_vec, axes=(0, 2, 1)
    )  # still general for n-dimensions
    # lags = [g - g.T for g in grid_vec]
    return lags


lags = get_lags(mask)

covMat2 = model(lags)


# stop


missing_point = missing_idxs[0]

xs = np.meshgrid(*(np.arange(0, m) for m in n), indexing="ij")
X = np.array(xs).reshape(2, np.prod(n)).T

X = X[np.where(m.flatten())]
d2 = np.sum((X[:, None] - X) ** 2, axis=2)

# stop
nugget_effect = model.nugget.value * (d2 == 0)
covMat = model.sigma.value**2 * np.exp(-0.5 * d2 / model.rho.value**2) + nugget_effect
# stop
covMat_inv = np.linalg.inv(covMat)

# lags = grid.lag_matrix
# covMat = model(lags)

lags2 = X - missing_point
acf = model(lags2.T)

# d2 = np.sum((X - missing_point)**2, axis=1)
# nugget_effect = model.nugget.value*np.all(d2 == 0, axis=0)
# acf = model.sigma.value ** 2 * np.exp(- 0.5*d2 / model.rho.value ** 2) + nugget_effect


# TODO: TRY WITH FULL COVMAT AND FULL OBSERVATIONS

weights = covMat_inv @ acf

mean1 = acf @ covMat_inv @ z[m]
mean2 = z[m] @ covMat_inv @ acf
mean3 = np.dot(weights, z[m])
print(mean1, mean2, mean3)

var1 = 1.1 - acf @ covMat_inv @ acf
var2 = 1.1 - np.dot(acf, weights)
print(var1, var2)


plt.imshow(covMat)
plt.show()

plt.plot(acf)
plt.show()
stop

ndim = len(n)

lags_ = lags - missing_point.reshape(-1, *[1] * ndim)

acf = model(lags_)
plt.imshow(acf, origin="lower")
plt.show()

stop

d = np.sqrt(sum(((lag - lag[(*missing_point,)]) ** 2 for i, lag in enumerate(lags))))

ind = np.unravel_index(np.argsort(d, axis=None), d.shape)
model
