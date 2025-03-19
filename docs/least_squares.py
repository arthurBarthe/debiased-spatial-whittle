from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("cupy")

np = BackendManager.get_backend()

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel, NuggetModel
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.least_squares import LeastSquareEstimator
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator

grid = RectangularGrid((512, 512))
model = SquaredExponentialModel(rho=12.0, sigma=2.0)
model = NuggetModel(model, nugget=1e-3)
sampler = SamplerOnRectangularGrid(model, grid)
data = sampler()

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid, periodogram)

model_est = SquaredExponentialModel(rho=10.0)
model_est_ = NuggetModel(model_est, nugget=1e-1)

debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

least_square = LeastSquareEstimator(periodogram, expected_periodogram)
least_square(data, model_est_)
print(model_est.rho, model_est.sigma, model_est_.nugget)
print(debiased_whittle(data, model_est_))


estimator = Estimator(debiased_whittle)
estimator(model_est_, data)
print(model_est.rho, model_est.sigma, model_est_.nugget)
print(debiased_whittle(data, model_est_))

model_est.sigma = 1
p = periodogram(data)
ep = expected_periodogram(model_est_)
print(np.sqrt(np.mean(p / ep)))

model_est.sigma = np.sqrt(np.mean(p / ep))
print(debiased_whittle(data, model_est_))
