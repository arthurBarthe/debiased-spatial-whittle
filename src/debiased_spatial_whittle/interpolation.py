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
        
        self.missing_points = np.argwhere(~self.grid.mask)
        
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
    
    def __call__(self, x: ndarray, params: ndarray):
        
        self.update_model_params(params)
        
        covMat = self.model(self.lags_wo_missing)   # Sigma_22
        covMat_inv = np.linalg.inv(covMat)          # TODO: bring outside function with update params
        
        xs = np.argwhere(self.grid.mask)
        acf = self.model((xs - x).T)
        
        weights  = covMat_inv @ acf
        
        pred_mean = np.dot(weights, z[self.grid.mask])
        # mean = acf @ covMat_inv @ self.z[self.grid.mask]
        pred_var = self.model(np.zeros(1)) - np.dot(acf, weights)
        return pred_mean, pred_var
        
    
    

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
model.rho = 10
model.sigma = 1
model.nugget=0.1
sampler = SamplerOnRectangularGrid(model, grid)
z_ = sampler()
z = z_ * mask

# plt.imshow(z, origin='lower', cmap='Spectral')
# plt.show()

interp = SimpleKriging(z, RectangularGrid(n, mask=m), model)
print(interp(np.array([4.,11.]), params=[10.,1.]))
