import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle
# from debiased_spatial_whittle.bayes_old import DeWhittle2


fftn = np.fft.fftn

np.random.seed(1252147)

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


params = np.log([10.,1.])

dw = DeWhittle(z, grid, SquaredExponentialModel())
eI = dw.expected_periodogram(np.exp(params))
# plt.imshow(fftshift(eI))
# plt.show()

print(dw(params))
print(dw.logpost(params))

from autograd import grad, hessian
ll = lambda x: dw(x)
print(grad(ll)(params))

niter=5000

dw.fit(None, prior=False)
dewhittle_post, A = dw.RW_MH(niter)
MLEs = dw.estimate_standard_errors_MLE(dw.res.x, monte_carlo=True, niter=500)
dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True)



whittle = Whittle(z, grid, SquaredExponentialModel())
whittle.fit(None, False)
whittle_post, A = whittle.RW_MH(niter)
# whittle.estimate_standard_errors_MLE(whittle.res.x, monte_carlo=True, niter=500)


title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle', 'Whittle']
plot_marginals([dewhittle_post, adj_dewhittle_post, whittle_post], params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))

