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

n_samples = 1000
sizes = [2**8, 2**10, 2**12, 2**14, 2**16]
estimates = []

for size in sizes:
    print(size)
    estimates_i = []
    grid = RectangularGrid((size,))
    model = FractionalGaussianNoise(hurst=0.7)
    sampler = SamplerOnRectangularGrid(model, grid)
    sampler.n_sims = 10

    periodogram = Periodogram()
    expected_periodogram = ExpectedPeriodogram(grid, periodogram)
    debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

    for i in range(n_samples):
        sample = sampler()

        hurst_est = np.linspace(0.55, 0.8, 100)
        model_est = FractionalGaussianNoise(hurst=hurst_est)
        lkh_values = debiased_whittle(sample, model_est)
        estimate = hurst_est[np.argmin(lkh_values)]
        estimates_i.append(estimate)
    estimates.append(np.array(estimates_i))

estimates = np.stack(estimates, 1)
rmse = np.sqrt(np.mean((estimates - 0.7) ** 2, axis=0))
print(rmse)
plt.figure()
plt.loglog(sizes, np.to_cpu(rmse), "--*", linewidth=5, markersize=15)
plt.xlabel("Time series length")
plt.ylabel("RMSE")
plt.show()
