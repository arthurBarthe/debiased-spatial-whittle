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

# np.random.seed(1252147)

n = (75, 75)
grid = RectangularGrid(n)
model = ExponentialModel()

sampler = SamplerOnRectangularGrid(model, grid)
# z = sampler()
z = np.loadtxt('sst_data.txt')

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(z, origin='lower', cmap='Spectral')
plt.show()



dw = DeWhittle(z, grid, ExponentialModel(), nugget=1e-10)
dw.fit(None, prior=False)
# stop
niter=5000

# TODO: cannot simulate z based on these params!

dewhittle_post, A = dw.RW_MH(niter)
MLEs = dw.estimate_standard_errors_MLE(dw.res.x, monte_carlo=True, niter=2000)
dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True)


title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle']
plot_marginals([dewhittle_post, adj_dewhittle_post], None, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))


# dw2 = DeWhittle(np.ones((150,150)), RectangularGrid((150,150)), ExponentialModel(), nugget=np.exp(dw.res.x[-1]))

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





# whittle = Whittle(z, grid, SquaredExponentialModel())
# whittle.fit(None, False)
# whittle_post, A = whittle.RW_MH(niter)
# whittle.estimate_standard_errors_MLE(whittle.res.x, monte_carlo=True, niter=500)




# gauss = Gaussian(z, grid, SquaredExponentialModel())
# gauss.fit(None, prior=False, approx_grad=True)
# print(gauss(params))

