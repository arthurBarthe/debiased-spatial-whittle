# We simulate from a 3d exponential covariance model and estimate its parameters from the simulation

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    SeparableModel,
)
from debiased_spatial_whittle.periodogram import (
    Periodogram,
    ExpectedPeriodogram,
    SeparableExpectedPeriodogram,
)
from debiased_spatial_whittle.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.simulation import (
    SamplerOnRectangularGrid,
    SamplerSeparable,
)
from debiased_spatial_whittle.utils import video_plot_3d

n = (256, 4096 * 2)

grid = RectangularGrid(n)

model_space = SquaredExponentialModel()
model_space.rho = 5
model_time = SquaredExponentialModel()
model_time.rho = 2

model = SeparableModel([model_space, model_time], dims=[(0,), (1,)])
model.merge_parameters(("sigma_0", "sigma_1"))
model.free_params["sigma_0 and sigma_1"].value = 1

sampler = SamplerOnRectangularGrid(model, grid)
z = sampler()
z = SampleOnRectangularGrid(grid, z)

# estimation
p = Periodogram()
ep = SeparableExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
e = Estimator(d, use_gradients=False)

model_space = SquaredExponentialModel()
model_time = SquaredExponentialModel()
model = SeparableModel([model_space, model_time], dims=[(0,), (1,)])
model.merge_parameters(("sigma_0", "sigma_1"))
print(model.free_params)
model.free_params["rho_0"].init_guess = 1
model.free_params["rho_1"].init_guess = 1
print(model.free_params.init_guesses)


def opt_callback(*args, **kargs):
    print(*args)
    print(**kargs)


print("start estimation")
print(e(model, z, opt_callback=opt_callback))
