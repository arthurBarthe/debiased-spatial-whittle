# Here we demonstrate a simple case of two random fields uniformely correlated and estimation via
# debiased whittle likelihood

import sys
import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('numpy')

from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, BivariateUniformCorrelation
from debiased_spatial_whittle.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerBUCOnRectangularGrid

corr = 0.9

g = RectangularGrid((64, 64), nvars=2)
m = ExponentialModel()
m.rho = 8
m.sigma = 1
m.nugget = 0.01
bvm = BivariateUniformCorrelation(m)
bvm.r_0 = corr
bvm.f_0 = 1.5
print(bvm)

s = SamplerBUCOnRectangularGrid(bvm, g)

data = s()
print(type(data))

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(data[..., 0], cmap='Spectral')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(data[..., 1], cmap='Spectral')
plt.show()

p = Periodogram()
p.taper = lambda x: x
p.fold = True

ep = ExpectedPeriodogram(g, p)

db = MultivariateDebiasedWhittle(p, ep)

rs = np.linspace(-0.95, 0.95, 100)
lkhs = np.zeros_like(rs)

for i, r in enumerate(rs):
    bvm.r_0 = r
    lkhs[i] = db(data, bvm)

plt.figure()
plt.plot(rs, lkhs, '-')
plt.show()

print(np.cov(data[..., 0].flatten(), data[..., 1].flatten()))

if input('Carry out optimization? (y/n) ... ') != 'y':
    sys.exit()

e = Estimator(db)
bvm.r_0 = None
bvm.rho_1 = None
bvm.f_0 = None
bvm.sigma_1 = None
print(e(bvm, data))

# we now carry out a test whose null hypothesis is that of zero-correlation
from debiased_spatial_whittle.hypothesis_tests import FixedParametersHT

bvm.r_0 = None
bvm.rho_1 = None
bvm.f_0 = None
bvm.sigma_1 = None
hypothesis_test = FixedParametersHT(bvm, dict(r_0=0.), db)
test_result = hypothesis_test(z=data)
print(test_result)

