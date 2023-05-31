import numpy as np
from debiased_spatial_whittle import fit, matern
import matplotlib.pyplot as plt
import scipy.io

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian



# load the topography data and standardize by the std
z = scipy.io.loadmat('Frederik53.mat')['topodata']
z = (z - np.mean(z)) / np.std(z)    # this is changed
n = z.shape

# plot
plt.figure()
plt.imshow(z, origin='lower', cmap='Spectral')
plt.show()

grid = RectangularGrid(n)


# init_guess = np.log([50, .5, 1, 1e-3])
# dw = DeWhittle(z, grid, MaternModel(), nugget=None)
# dw.fit(init_guess, prior=False, approx_grad=True)
# stop



model = MaternModel()                # try exponential model
model.nu = 0.8860611533657291
nugget = 0.0012308293585964476
dw = DeWhittle(z, grid, model, nugget=nugget)
init_guess = np.log([8.33, 1.786])
dw.fit(init_guess, approx_grad=True, prior=False)
# print(dw.propcov)

niter=2000
dewhittle_post, A = dw.RW_MH(niter, acceptance_lag=100)
# stop

MLEs = dw.sim_MLEs(np.exp(dw.res.x), niter=500, approx_grad=True)
dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True, acceptance_lag=100)


title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle']
plot_marginals([dewhittle_post, adj_dewhittle_post], None, title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))



sim_z = dw.sim_z(np.exp(dw.res.x))

fig, ax = plt.subplots(2,1, figsize=(40,10))
ax[0].set_title('Sea Temperature Data', fontsize=22)
im1 = ax[0].imshow(z, cmap='Spectral', origin='lower')
fig.colorbar(im1, ax=ax[0])

ax[1].set_title('simulated', fontsize=22)
im2 = ax[1].imshow(sim_z, cmap='Spectral', origin='lower')
fig.colorbar(im2, ax=ax[1])
    
fig.tight_layout()
plt.show()
