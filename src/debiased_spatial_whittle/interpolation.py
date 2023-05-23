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




class SimpleKriging:
    '''class for interpolation via simple Kriging'''    
    
    def __init__(self, z: np.ndarray, grid: RectangularGrid, model: CovarianceModel):
        
        self._z = z
        self.grid = grid
        self.model = model
        
        self._lags_wo_missing = self.lags_without_missing()
        
    @property
    def lags_wo_missing(self):
        return self._lags_wo_missing 
        
    def lags_without_missing(self):
        mask = self.grid.mask.astype(bool)
        flat_idxs = np.where(mask.flatten())
        xs = [np.arange(s, dtype=np.int64) for s in self.grid.n]
        grid = np.meshgrid(*xs, indexing='ij')
        grid_vec = [g.reshape((-1, 1))[flat_idxs] for g in grid]
        lags = [g - g.T for g in grid_vec]
        return np.array(lags)
    
    

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

interp = SimpleKriging(z, RectangularGrid(n, mask=mask), model)
