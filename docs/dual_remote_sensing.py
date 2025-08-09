import matplotlib.pyplot as plt

from debiased_spatial_whittle.models import SquaredExponentialModel, DualRemoteSensing
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import MultivariateSamplerOnRectangularGrid

from debiased_spatial_whittle.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import MultivariateDebiasedWhittle, Estimator

grid = RectangularGrid((64, 64), nvars=2)
latent_model = SquaredExponentialModel(rho=5.0)
model = DualRemoteSensing(latent_model, sigma1=0.1, sigma2=0.2)

sampler = MultivariateSamplerOnRectangularGrid(model, grid, p=2)
s = sampler()
print(s.shape)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(s[..., 0])
plt.subplot(1, 2, 2)
plt.imshow(s[..., 1])
plt.show()

# inference
latent_model = SquaredExponentialModel(rho=1.0)
model = DualRemoteSensing(latent_model, sigma1=0.01, sigma2=0.01)
model.set_param_bounds(dict(sigma1=(0.001, 1.0), sigma2=(0.001, 1.0)))

periodogram = Periodogram()
dbw = MultivariateDebiasedWhittle(periodogram, ExpectedPeriodogram(grid, periodogram))
estimator = Estimator(dbw)
estimator(model, s, opt_callback=lambda *args, **kwargs: print(*args, **kwargs))

# predict
import numpy as np

xs = np.stack(np.mgrid[:64, :64], -1).reshape(-1, 2)
ys = model.predict(
    (xs, xs), (s[..., 0].reshape((-1, 1)), s[..., 1].reshape((-1, 1))), xs
)
ys = ys.reshape((64, 64))
plt.figure()
plt.imshow(ys)
plt.show()
