# We simulate from a 3d exponential covariance model and estimate its parameters from the simulation

# ##Imports
import matplotlib.pyplot as plt
from IPython.display import HTML
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.models.univariate import (
    SquaredExponentialModel,
    NuggetModel,
)
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.utils import video_plot_3d

# ##Grid and model specification

n = (32, 32, 256)
grid = RectangularGrid(n)

model = SquaredExponentialModel(rho=8.0, sigma=0.8)
model = NuggetModel(model, nugget=0.0001)
# ##Sample generation

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()


anim = video_plot_3d(z, get_title=lambda i: "", cmap="Spectral")
plt.close()
HTML(anim.to_jshtml())

# ##Inference

p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
e = Estimator(d, use_gradients=False)

model = SquaredExponentialModel(rho=2.0, sigma=1)
model = NuggetModel(model, nugget=0.0001)
model.fix_parameter("nugget")


def opt_callback(*args, **kargs):
    print(*args)
    print(**kargs)


print("start estimation")
e(model, z, opt_callback=opt_callback)
