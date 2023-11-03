# We simulate from a 3d exponential covariance model and estimate its parameters from the simulation
import numpy as np
from matplotlib import pyplot as plt

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, SeparableModel
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid, SamplerSeparable
from debiased_spatial_whittle.utils import video_plot_3d

n = (128, 128, 32)

grid = RectangularGrid(n)
grid_space = RectangularGrid(n[:2])

model_space = SquaredExponentialModel()
model_space.rho = 6
model_time = SquaredExponentialModel()
model_time.rho = 3

model_std = SquaredExponentialModel()
model_std.rho = 4
model_std.sigma = 1
sampler_std = SamplerOnRectangularGrid(model_std, grid_space)
g = np.exp(sampler_std() - 2)
g = np.expand_dims(g, -1)

plt.figure()
plt.imshow(g)
plt.show()

model = SeparableModel([model_space, model_time], dims=[(0, 1), (2, )])
model.merge_parameters(('sigma_0', 'sigma_1'))
model.free_params['sigma_0 and sigma_1'].value = 1

sampler = SamplerOnRectangularGrid(model, grid)
#sampler = SamplerSeparable(model, grid)
z = sampler()

y = z * g

plt.figure()
plt.imshow(y[..., 0])
plt.show()

anim = video_plot_3d(y)
plt.show()

# estimation of g and z
g_hat = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
z_hat = y / g_hat

plt.figure()
plt.imshow(g_hat / g)
plt.title('g hat')
plt.show()


# estimation
p = Periodogram()
ep = ExpectedPeriodogram(grid, p)
d = DebiasedWhittle(p, ep)
e = Estimator(d, use_gradients=False)

model_space = SquaredExponentialModel()
model_time = SquaredExponentialModel()
model = SeparableModel([model_space, model_time], dims=[(0, 1), (2, )])
model.merge_parameters(('sigma_0', 'sigma_1'))
print(model.free_params)
model.free_params['rho_0'].init_guess = 1
model.free_params['rho_1'].init_guess = 1
print(model.free_params.init_guesses)

def opt_callback(*args, **kargs):
    print(*args)
    print(**kargs)

print('start estimation')
print(e(model, z_hat, opt_callback=opt_callback))