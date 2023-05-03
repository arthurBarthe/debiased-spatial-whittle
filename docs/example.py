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
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian
# from debiased_spatial_whittle.bayes_old import DeWhittle2


fftn = np.fft.fftn

np.random.seed(1252147)

n = (64, 64)
rho, sigma, nugget = 10., np.sqrt(1.), 0.1

grid = RectangularGrid(n)
model = ExponentialModel()
model.rho = rho
model.sigma = sigma
model.nugget = nugget


per = Periodogram()
# ep = ExpectedPeriodogram(grid, per)
# db = DebiasedWhittle(per, ep)

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

I = per(z)
# eI = fftshift(ep(model))
# plt.imshow(fftshift(I))
# plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(z, origin='lower', cmap='Spectral')
plt.show()

# stop
params = np.log([rho,sigma])

dw = DeWhittle(z, grid, ExponentialModel(), nugget=nugget)
# stop

# eI = dw.expected_periodogram(np.exp(params))
# plt.imshow(fftshift(eI))
# plt.show()

# print(dw(params))
# print(dw.logpost(params))

from autograd import grad, hessian
ll = lambda x: dw(x)
# print(grad(ll)(params))

niter=10000

dw.fit(None, prior=False)
dewhittle_post, A = dw.RW_MH(niter)
MLEs = dw.estimate_standard_errors_MLE(dw.res.x, monte_carlo=True, niter=2000)
dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True)

title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle']
plot_marginals([dewhittle_post, adj_dewhittle_post], params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))

stop

whittle = Whittle(z, grid, ExponentialModel(), nugget=nugget)
whittle.fit(None, False)
whittle_post, A = whittle.RW_MH(niter)
whittle.estimate_standard_errors_MLE(whittle.res.x, monte_carlo=True, niter=200)
whittle.prepare_curvature_adjustment()
adj_whittle_post, A = whittle.RW_MH(niter, adjusted=True)


title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle', 'Whittle', 'adj Whittle']
plot_marginals([dewhittle_post, adj_dewhittle_post, whittle_post, adj_whittle_post], params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))

stop

# gauss = Gaussian(z, grid, SquaredExponentialModel())
# gauss.fit(None, prior=False, approx_grad=True)
# print(gauss(params))




dfs = list(range(5,15+1)) + list(range(20,55,5)) + [9999]
# dfs = [5,6]
MLEs = [dw.sim_MLEs(params, 500, t_random_field=True, nu=df) for df in dfs]


title = 'DeWhittle t-random field MLE distribution'
legend_labels = [rf'$\nu={nu}$' for nu in dfs]
plot_marginals(MLEs, params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2), cmap='Spectral')

