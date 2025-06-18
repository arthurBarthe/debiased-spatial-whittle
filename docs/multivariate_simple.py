# Here we demonstrate a simple case of two random fields uniformely correlated and estimation using the
# multivariate version of the Debiased Whittle likelihood

# ##Imports

import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")

from debiased_spatial_whittle.models import (
    ExponentialModel,
    Matern32Model,
    BivariateUniformCorrelation,
)
from debiased_spatial_whittle.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import (
    SamplerBUCOnRectangularGrid,
    MultivariateSamplerOnRectangularGrid,
)

# ##Grid specification

# Note that we use the argument nvars=2 to specify that we observe a bivariate random field

g = RectangularGrid((128, 128), nvars=2)


# ##Model specification

# we set the correlation to 0.8

m = Matern32Model(rho=8, sigma=1)
bvm = BivariateUniformCorrelation(m, r=0.8, f=1.5)
bvm

# ##Sample generation

s = MultivariateSamplerOnRectangularGrid(bvm, g, p=2)
data = s()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(data[..., 0], cmap="inferno")
ax = fig.add_subplot(1, 2, 2)
ax.imshow(data[..., 1], cmap="inferno")
plt.show()

# ##Profile likelihood plot for the correlation parameter

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
plt.xlabel("correlation coefficient")
plt.ylabel("profile negative log-likelihood")
plt.show()

# ##Inference

e = Estimator(db)
m = ExponentialModel(rho=1, sigma=1)
bvm = BivariateUniformCorrelation(m, r=0.0, f=1.0)
bvm.set_param_bounds(dict(r=(-0.95, 0.95)))
e(bvm, data, opt_callback=lambda *args, **kwargs: print(bvm.r))
print(bvm.r)
