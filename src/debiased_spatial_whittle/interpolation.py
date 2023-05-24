import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import CovarianceModel, ExponentialModel, SquaredExponentialModel, MaternModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram, compute_ep
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian
# from debiased_spatial_whittle.bayes_old import DeWhittle2


ndarray = np.ndarray

class SimpleKriging:
    '''Interpolation via simple Kriging'''    
    
    def __init__(self, z: np.ndarray, grid: RectangularGrid, model: CovarianceModel):
        
        self.z = z    # TODO: change to property
        grid.mask = grid.mask.astype(bool)
        self.grid = grid
        self.model = model

        self.xs = np.argwhere(self.grid.mask)
        self.missing_xs = np.argwhere(~self.grid.mask)
        
        self._lags_wo_missing = self.lags_without_missing()
        
    @property
    def lags_wo_missing(self):
        return self._lags_wo_missing 
        
    def lags_without_missing(self):
        grid_vec = np.argwhere(self.grid.mask)[None].T
        lags = grid_vec - np.transpose(grid_vec, axes=(0,2,1))   # still general for n-dimensions
        return np.array(lags)
    
    def update_model_params(self, params: ndarray) -> None:
        free_params = self.model.params        
        updates = dict(zip(free_params.names, params))
        free_params.update_values(updates)
        return
    
    def compute_inv_covmat(self, params: ndarray):
        self.update_model_params(params)
        
        covMat = self.model(self.lags_wo_missing)   # Sigma_22
        covMat_inv = np.linalg.inv(covMat)          # TODO: bring outside function with update params
        return covMat_inv
    
    @staticmethod
    def format_arr(x):
        x = np.array(x)
        if x.ndim == 1:
            xs = x[None].copy()
            
        elif x.ndim ==2:
            xs = x.copy()
        else: 
            raise TypeError('x must be array of point(s), < 2 dim')
        return xs

    
    def __call__(self, x: ndarray, params: ndarray):

        covMat_inv = self.compute_inv_covmat(params)
        # self.update_model_params(params)
        
        xs = self.format_arr(x)        
        
        pred_means = np.zeros(len(xs))
        pred_vars  = np.zeros(len(xs))
        for i,point in enumerate(xs):
        
            acf = self.model((self.xs - point).T)
            
            weights  = covMat_inv @ acf
            
            pred_means[i] = np.dot(weights, self.z[self.grid.mask])
            # mean = acf @ covMat_inv @ self.z[self.grid.mask]
            
            pred_vars[i] = self.model(np.zeros(1)) - np.dot(acf, weights)
            
        return pred_means, pred_vars
    
    def bayesian_prediction(self, x: ndarray, posterior_samples: ndarray):

        xs = self.format_arr(x)
        posterior_samples = self.format_arr(posterior_samples)
        
        m = len(xs)
        ndraws = len(posterior_samples)
        xs_pred_draws = np.zeros((ndraws, m))
        for i, sample in enumerate(posterior_samples):
            print(f'\rComputed {i+1} out of {ndraws} posterior draws', end='')
            pred_means, pred_vars = self(x, np.exp(sample))
            xs_pred = pred_means + np.sqrt(pred_vars)*np.random.randn(m)
            xs_pred_draws[i] = xs_pred
        print()
        return xs_pred_draws
            


# TODO: make test

np.random.seed(1252147)

n = (64, 64)

mask = np.ones(n)

n_missing = 10
missing_idxs = np.random.randint(n[0], size=(n_missing,2))
mask[tuple(missing_idxs.T)] = 0.
m = mask.astype(bool)

plt.imshow(mask, cmap='Greys', origin='lower')
plt.show()

grid = RectangularGrid(n)

model = SquaredExponentialModel()
model.rho    = 10
model.sigma  = 1
model.nugget = 1e-5
sampler = SamplerOnRectangularGrid(model, grid)
z_ = sampler()
z = z_ * mask

params = np.log([10.,1.])

grid = RectangularGrid(n, mask=m)
dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1)
dw.fit(None, prior=False)

interp = SimpleKriging(z, RectangularGrid(n, mask=m), model)
pred_means, pred_vars = interp(interp.missing_xs, params=np.exp(dw.res.x))

print(z_[~m].round(3), pred_means.round(3), sep='\n')

z[~m] = pred_means

fig, ax = plt.subplots(1,2, figsize=(20,15))
ax[0].set_title('original', fontsize=22)
im1 = ax[0].imshow(z_, cmap='Spectral', origin='lower')
fig.colorbar(im1, shrink=.5, ax=ax[0])

ax[1].set_title('interpolated', fontsize=22)
im2 = ax[1].imshow(z, cmap='Spectral', origin='lower')
fig.colorbar(im2, shrink=.5, ax=ax[1])
fig.tight_layout()

dewhittle_post, A = dw.RW_MH(200)  # unadjusted
preds = interp.bayesian_prediction(interp.missing_xs, dewhittle_post)
plot_marginals(preds.T, shape=(2,5), truths=z_[~m], title='posterior predictive densities')
