# We simulate from a 3d exponential covariance model and estimate its parameters from the simulation

# ##Imports

from IPython.display import HTML
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.utils import video_plot_3d

# ##Grid and model specification

n = (32, 32, 256)
grid = RectangularGrid(n)

model = SquaredExponentialModel()
model.sigma = 1
model.rho = 8
model.nugget = 0.01

# ##Sample generation

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()


anim = video_plot_3d(z, get_title=lambda i: "", cmap="Spectral")
HTML(anim.to_html5_video())

# ##Inference

p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
e = Estimator(d, use_gradients=False)

model = SquaredExponentialModel()
model.nugget = None
print(model.free_params)
print(model.free_params.init_guesses)


def opt_callback(*args, **kargs):
    print(*args)
    print(**kargs)


print("start estimation")
print(e(model, z, opt_callback=opt_callback))
