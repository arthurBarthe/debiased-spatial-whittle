# Here we demonstrate a simple case of two random fields uniformely correlated and estimation using the
# multivariate version of the Debiased Whittle likelihood

# ##Imports

import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("numpy")

from debiased_spatial_whittle.models import (
    ExponentialModel,
    BivariateUniformCorrelation,
)
from debiased_spatial_whittle.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerBUCOnRectangularGrid

# ##Grid specification

# Note that we use the argument nvars=2 to specify that we observe a bivariate random field

g = RectangularGrid((128, 128), nvars=2)


# ##Model specification

# we set the correlation to 0.9

m = ExponentialModel(rho=8, sigma=1)
bvm = BivariateUniformCorrelation(m)
bvm.r = 0.8
bvm.f = 1.5
print(bvm)

# ##Sample generation

s = SamplerBUCOnRectangularGrid(bvm, g)
data = s()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(data[..., 0], cmap="Spectral")
ax = fig.add_subplot(1, 2, 2)
ax.imshow(data[..., 1], cmap="Spectral")
plt.show()

# ##Profile likelihood plot for the correlation parameter

p = Periodogram()
p.fold = True

ep = ExpectedPeriodogram(g, p)
print(ep(bvm).shape)
db = MultivariateDebiasedWhittle(p, ep)

rs = np.linspace(-0.95, 0.95, 100)
lkhs = np.zeros_like(rs)

for i, r in enumerate(rs):
    print(i)
    bvm.r = r
    lkhs[i] = db(data, bvm)

plt.figure()
plt.plot(rs, lkhs, "-")
plt.show()

print(np.cov(data[..., 0].flatten(), data[..., 1].flatten()))


# ##Inference

e = Estimator(db)
bvm.r_0 = None
bvm.rho_1 = None
bvm.f_0 = None
bvm.sigma_1 = None
print(e(bvm, data))

# ##Hypothesis test of zero-correlation

from debiased_spatial_whittle.hypothesis_tests import FixedParametersHT

bvm.r_0 = None
bvm.rho_1 = None
bvm.f_0 = None
bvm.sigma_1 = None
hypothesis_test = FixedParametersHT(bvm, dict(r_0=0.0), db)
test_result = hypothesis_test(z=data)
print(test_result)
