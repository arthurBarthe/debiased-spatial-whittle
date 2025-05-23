import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("cupy")

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel
from debiased_spatial_whittle.tsmodels import FractionalGaussianNoise
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import DebiasedWhittle

np = BackendManager.get_backend()

grid = RectangularGrid((4096 * 8,), (1,))
model = FractionalGaussianNoise(hurst=0.6)
sampler = SamplerOnRectangularGrid(model, grid)

sample = sampler()

plt.figure()
plt.plot(np.to_cpu(sample))
plt.show()

# inference
periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)

debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

hurst_est = np.linspace(0.51, 0.9, 20)
model_est = FractionalGaussianNoise(hurst=hurst_est)
lkh_values = debiased_whittle(sample, model_est)

plt.figure()
plt.plot(np.to_cpu(hurst_est), np.to_cpu(lkh_values), "-*")
plt.show()
