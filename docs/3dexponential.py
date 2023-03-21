# We simulate from a 3d exponential covariance model and estimate its parameters from the simulation

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, SeparableModel
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SamplerSeparable
from debiased_spatial_whittle.utils import video_plot_3d

n = (128, 128, 128)

grid = RectangularGrid(n)
model = ExponentialModel()
model.sigma = 1
model.rho = 3

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()

video_plot_3d(z)

# estimation
p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
e = Estimator(d, use_gradients=False)

model = ExponentialModel()
print(model.free_params)
print(model.free_params.init_guesses)

def opt_callback(*args, **kargs):
    print(*args)
    print(**kargs)

print('start estimation')
print(e(model, z, opt_callback=opt_callback))