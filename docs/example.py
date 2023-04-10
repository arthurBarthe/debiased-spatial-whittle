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
from debiased_spatial_whittle.Likelihoods import DebiasedWhittle2


fftn = np.fft.fftn

# np.random.seed(1252149)

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

dw.fit(None, prior=False)

dw.fit(None, prior=False, label='whittle')


dw2 = DebiasedWhittle2(z, grid, SquaredExponentialModel())
dw2.fit(None, prior=False)

from debiased_spatial_whittle.bayes2.funcs import gradf

print(gradf(np.ones(5)))

# stop


dw.fit(None, prior=False)
niter=5000
dewhittle_post, A = dw.RW_MH(niter)

MLEs = dw.estimate_standard_errors(dw.res.x, monte_carlo=True, niter=500,  const='whittle')

dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, posterior_name='adj deWhittle', propcov=dw.adj_deWhittle_propcov)

title = 'comparison'
legend_labels = ['deWhittle', 'adj deWhittle', 'MLE distribution']
plot_marginals([dewhittle_post, adj_dewhittle_post, MLEs], params, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))
