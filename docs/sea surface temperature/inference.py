import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian
# from debiased_spatial_whittle.bayes_old import DeWhittle2


fftn = np.fft.fftn
fftfreq = np.fft.fftfreq
fftshift = np.fft.fftshift

# np.random.seed(1252147)

n = (75, 75)
grid = RectangularGrid(n)

z = np.loadtxt('sst_data.txt')

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(z, origin='lower', cmap='Spectral')
plt.show()


params = np.log([10, 1., 0.733])
model = MaternCovarianceModel()
# model.nu = 0.733

dw = DeWhittle(z, grid, model, nugget=1e-10)
dw.fit(params, prior=False, approx_grad=True)

niter=1000
dewhittle_post, A = dw.RW_MH(niter, acceptance_lag=1000)

# stop

MLEs = dw.sim_MLEs(np.exp(dw.res.x), niter=500, approx_grad=True)
dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True, acceptance_lag=100)


title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle']
plot_marginals([dewhittle_post, adj_dewhittle_post], None, title, [r'log$\rho$', r'log$\sigma$', r'log$\nu$'], legend_labels, shape=(1,3))


sim_z = dw.sim_z(np.exp(dw.res.x))

fig, ax = plt.subplots(1,2, figsize=(20,15))
ax[0].set_title('Sea Temperature Data', fontsize=22)
im1 = ax[0].imshow(z, cmap='Spectral', origin='lower', extent = [211.125, 230, 20 ,38.875])
fig.colorbar(im1, shrink=.5, ax=ax[0])

ax[1].set_title('simulated', fontsize=22)
im2 = ax[1].imshow(sim_z, cmap='Spectral', origin='lower', extent = [211.125, 230, 20 ,38.875])
fig.colorbar(im2, shrink=.5, ax=ax[1])

for i in range(2):
    ax[i].set_xlabel('Longitude', fontsize=16)
    ax[i].set_ylabel('Latitude',  fontsize=16)
    
    ax[i].set_xticks(np.arange(215,235, 5))
    ax[i].set_yticks(np.arange(20 ,38.875, 5))
    
fig.tight_layout()
plt.show()

model = MaternModel()
model.nu = 0.733
N = (500,500)
dw_ = DeWhittle(z,RectangularGrid(N), model=model, nugget=1e-10)


q05, q95 = np.quantile(adj_dewhittle_post, q=[0.05,0.95], axis=0)
post_mean = np.mean(adj_dewhittle_post, axis=0)

f_q05 = fftshift(dw_.expected_periodogram(np.exp(q05)))
f_q95 = fftshift(dw_.expected_periodogram(np.exp(q95)))
f_mean = fftshift(dw_.expected_periodogram(np.exp(post_mean)))

freq_grid = np.meshgrid(*(fftshift(2*np.pi*fftfreq(_n)) for _n in N), indexing='ij')         # / (delta*n1)?
omegas_grid = np.sqrt(sum(grid**2 for grid in freq_grid))
omegas = np.diag(omegas_grid)[:N[0]//2]
fig = plt.figure(figsize=(10,7))
plt.plot(omegas, np.diag(f_q05)[:N[0]//2], '--', c='#1f77b4')
plt.plot(omegas, np.diag(f_mean)[:N[0]//2], c= '#1f77b4')
plt.plot(omegas, np.diag(f_q95)[:N[0]//2], '--', c= '#1f77b4')
plt.fill_between(omegas, np.diag(f_q05)[:N[0]//2], np.diag(f_q95)[:N[0]//2], color='#1f77b4', alpha=0.25)
# plt.xlim([-0.1,1.])
plt.show()

from debiased_spatial_whittle.diagnostics import DiagnosticTest
test = DiagnosticTest(dw.I, dw.expected_periodogram(np.exp(post_mean)))
test()

fig, ax = plt.subplots(figsize=(15,7))
im = plt.imshow(fftshift(test.residuals), cmap='Spectral', origin='lower', extent=[-np.pi,np.pi]*2)
plt.title('residuals', fontsize=22)
plt.colorbar(im, ax=ax, pad=0.01)
plt.show()


# whittle = Whittle(z, grid, SquaredExponentialModel())
# whittle.fit(None, False)
# whittle_post, A = whittle.RW_MH(niter)
# whittle.estimate_standard_errors_MLE(whittle.res.x, monte_carlo=True, niter=500)




# gauss = Gaussian(z, grid, SquaredExponentialModel())
# gauss.fit(None, prior=False, approx_grad=True)
# print(gauss(params))

