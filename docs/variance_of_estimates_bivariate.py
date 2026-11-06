import torch
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.models.bivariate import BivariateUniformCorrelation
from debiased_spatial_whittle.inference.multivariate_periodogram import Periodogram
from debiased_spatial_whittle.inference.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import MultivariateDebiasedWhittle, Estimator
from debiased_spatial_whittle.sampling.simulation import MultivariateSamplerOnRectangularGrid

_ = torch.manual_seed(0)
model = ExponentialModel(rho=torch.tensor(12.), sigma=torch.tensor(1.))
model = BivariateUniformCorrelation(model, r=0.2, f=1.3)
periodogram = Periodogram()
grid = RectangularGrid((67, 192), nvars=2)
ep = ExpectedPeriodogram(grid, periodogram)
dbw = MultivariateDebiasedWhittle(periodogram, ep)
jmat = dbw.jmatrix_sample(model, n_sims=1000)
predicted_covarance_matrix = dbw.variance_of_estimates(model, jmat=jmat)


sampler = MultivariateSamplerOnRectangularGrid(model, grid, p=2)
estimator = Estimator(dbw)

def get_estimation_model():
    model = ExponentialModel(rho=torch.tensor(12.), sigma=torch.tensor(1.))
    model = BivariateUniformCorrelation(model, r=0.2, f=1.3)
    return model

estimates = []
for i in range(1000):
    sample = sampler()
    model_est = get_estimation_model()
    estimator(model_est, sample)
    print(model_est)
    estimates.append(model_est.parameters)

estimates = torch.asarray(estimates)

d = torch.diagonal(predicted_covarance_matrix)
d = d.reshape((-1, 1))

print(d.flatten())
print(torch.diagonal(torch.cov(estimates.T)))

print(predicted_covarance_matrix / torch.sqrt((d @ d.T)))
print(torch.corrcoef(estimates.T))