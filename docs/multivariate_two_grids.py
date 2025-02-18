# In this example, the two random fields sampling locations do not overlap. Yet we are able to estimate
# the correlation parameter. Note that we would not be able to do so if the lengthscale parameter were close to zero.

# ##Imports

import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")

from debiased_spatial_whittle.models import (
    SquaredExponentialModel,
    BivariateUniformCorrelation,
)
from debiased_spatial_whittle.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerBUCOnRectangularGrid

# ##Grid specification

g = RectangularGrid((128, 128), nvars=2)
g.mask = np.random.rand(*g.mask.shape) > 0.2
x, y = g.grid_points
g.mask[..., 0] = np.mod(x, 40) <= 20
g.mask[..., 1] = np.mod(x, 40) > 20

# ##Model definition

m = SquaredExponentialModel()
m.rho = 12
m.sigma = 1
m.nugget = 0.01
bvm = BivariateUniformCorrelation(m)
bvm.r = 0.5
bvm.f = 1.9
print(bvm)

# ##Sample generation

s = SamplerBUCOnRectangularGrid(bvm, g)

data = s()
print(type(data))

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(data[..., 0], cmap="Spectral")
ax = fig.add_subplot(1, 2, 2)
ax.imshow(data[..., 1], cmap="Spectral")
plt.show()

# ##Inference

p = Periodogram()
p.fold = True

ep = ExpectedPeriodogram(g, p)
db = MultivariateDebiasedWhittle(p, ep)

rs = np.linspace(-0.95, 0.95, 100)
lkhs = np.zeros_like(rs)

for i, r in enumerate(rs):
    bvm.r = r
    lkhs[i] = db(data, bvm)

plt.figure()
plt.plot(rs, lkhs, "-")
plt.show()
