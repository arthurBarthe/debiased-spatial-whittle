import numpy as np
from debiased_spatial_whittle import fit, matern
import matplotlib.pyplot as plt
import scipy.io

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.interpolation import SimpleKriging
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian

np.random.seed(15344352)

# load the topography data and standardize by the std
realdata = scipy.io.loadmat('Frederik53.mat')['topodata']
realdata = (realdata - np.mean(realdata)) / np.std(realdata)    # this is changed
n = realdata.shape

# plot
plt.figure()
plt.imshow(realdata, origin='lower', cmap='Spectral')
plt.show()



# frequency mask corresponding to the data processing
from numpy.fft import fftfreq
n1, n2 = realdata.shape
x, y = np.meshgrid(fftfreq(n1) * 2 * np.pi, fftfreq(n2) * 2 * np.pi, indexing='ij')
freq_norm = np.sqrt(x ** 2 + y ** 2)
frequency_mask = freq_norm < np.pi
print(realdata.shape, frequency_mask.shape)

plt.figure()
plt.imshow(frequency_mask)
plt.show()


mask = np.ones(n, dtype=bool)

print(mask.shape)

# picking missing points within the frequency mask
n_missing = 10
full_obs_grid = np.argwhere(frequency_mask)
missing_idxs = np.random.randint(len(full_obs_grid), size=n_missing)
missing_points = full_obs_grid[missing_idxs]
mask[tuple(missing_points.T)] = False

plt.figure()
plt.imshow(mask)
plt.show()
# stop

z = realdata * mask

grid = RectangularGrid(n, mask=mask)

# plt.imshow(z)
# stop
init_guess = np.log([1., 1., 1.])
dw = DeWhittle(z, grid, MaternCovarianceModel(), nugget=1e-10)
dw.frequency_mask = frequency_mask
dw.fit(init_guess, prior=False, approx_grad=True)

# import matplotlib as mpl
# bounds = np.quantile(z, np.linspace(0,1,500)) + 0.7
# # bounds = np.linspace(0,1,500)
# cm = plt.get_cmap('Spectral')
# rbga = cm(bounds)
# cmap = mpl.colors.ListedColormap(rbga, name='myColorMap', N=rbga.shape[0])

# model = MaternModel()                # try exponential model
# model.nu = 0.8860611533657291
# nugget = 0.0012308293585964476
# dw = DeWhittle(z, grid, model, nugget=nugget)
# dw.frequency_mask = frequency_mask


# init_guess = np.log([50.,1.])
# dw.fit(init_guess, approx_grad=True, prior=False)
# print(dw.propcov)

# stop

niter=2000
dewhittle_post, A = dw.RW_MH(niter, acceptance_lag=100)
# stop

_MLEs = dw.sim_MLEs(np.exp(dw.res.x), niter=500, approx_grad=True)
MLEs = _MLEs[np.exp(_MLEs[:,0])<300]    # TODO: filter outliers
dw.MLEs = MLEs
dw.MLEs_cov = np.cov(MLEs.T)
dw.prepare_curvature_adjustment()
adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True, acceptance_lag=100)


title = 'posterior comparisons'
legend_labels = ['deWhittle', 'adj deWhittle']
plot_marginals([dewhittle_post, adj_dewhittle_post], None, title, [r'log$\rho$', r'log$\sigma$', r'log$\nu$'], legend_labels, shape=(1,3))



sim_z = dw.sim_z(np.exp(dw.res.x))

fig, ax = plt.subplots(2,1, figsize=(20,10))
ax[0].set_title('Sea Temperature Data', fontsize=22)
im1 = ax[0].imshow(z, cmap='Spectral', origin='lower')
fig.colorbar(im1, ax=ax[0])

ax[1].set_title('simulated', fontsize=22)
im2 = ax[1].imshow(sim_z, cmap='Spectral', origin='lower')
fig.colorbar(im2, ax=ax[1])
    
fig.tight_layout()
plt.show()



interp = SimpleKriging(z, grid, MaternCovarianceModel())
approx_preds = interp.approx_bayesian_prediction(interp.missing_xs, adj_dewhittle_post, n_closest=100)
plot_marginals(approx_preds.T, shape=(2,5), truths=realdata[~mask], title='posterior predictive densities')


