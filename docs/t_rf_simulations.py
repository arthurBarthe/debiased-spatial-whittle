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


fftn = np.fft.fftn

np.random.seed(1252147)

rho, sigma, nugget = 10., np.sqrt(1.), 0.1

model = ExponentialModel()
model.rho = rho
model.sigma = sigma
model.nugget = nugget


params = np.log([rho,sigma])

grid_sizes = [(2**i,)*2 for i in range(5,8+1)]
dfs = list(range(5,15+1)) + list(range(20,55,5)) + [9999]

# grid_sizes = [(2**i,)*2 for i in range(5,6+1)]
# dfs = [5,6]

MLEs = {}
for i, n in enumerate(grid_sizes):
    
    grid = RectangularGrid(n)
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    dw = DeWhittle(z, grid, ExponentialModel(), nugget=nugget)
    MLEs[f'{n}'] = [dw.sim_MLEs(params, niter=2000, t_random_field=True, nu=df) for df in dfs]


