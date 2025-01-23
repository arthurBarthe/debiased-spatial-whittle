import autograd.numpy as np
from autograd import grad
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import inv
from autograd.numpy import ndarray

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.likelihood import DebiasedWhittle, Estimator
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import (
    Periodogram,
    ExpectedPeriodogram,
    compute_ep,
)
from debiased_spatial_whittle.spatial_kernel import spatial_kernel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian


n = (64, 64)
rho, sigma, nugget = 10.0, np.sqrt(1.0), 0.1

grid = RectangularGrid(n)
model = SquaredExponentialModel()
model.rho = rho
model.sigma = sigma
model.nugget = nugget


sampler = SamplerOnRectangularGrid(model, grid)

params = np.log([rho, sigma])

# stop


n_datasets = 10
mcmc_niter = 1000
mle_niter = 200
acceptance_lag = mcmc_niter + 1
d = len(params)


def make_quantiles(alphas: list | ndarray):
    quant_list = []
    for alpha in alphas:
        if alpha > 1:
            alpha /= 100
        quant_list.append(alpha / 2)
        quant_list.append(1 - (alpha / 2))
    return quant_list


alphas = np.arange(5, 100, 5)
alphas_list = [alpha for alpha in alphas for _ in (0, 1)] * 8
quantiles = make_quantiles(alphas)
n_q = len(quantiles)

dewhittle_post_quantiles = np.zeros((n_datasets, d * n_q))
adj_dewhittle_post_quantiles = np.zeros((n_datasets, d * n_q))
whittle_post_quantiles = np.zeros((n_datasets, d * n_q))
adj_whittle_post_quantiles = np.zeros((n_datasets, d * n_q))

print(f"True Params:  {np.round(np.exp(params),3)}")
for i in range(n_datasets):
    print(i + 1, end=":\n")

    z = sampler()

    for likelihood in [DeWhittle, Whittle]:  # ,Whittle
        ll = likelihood(z, grid, SquaredExponentialModel(), nugget=nugget)
        ll.fit(None, prior=False, print_res=False)
        post, A = ll.RW_MH(mcmc_niter, acceptance_lag=acceptance_lag)
        MLEs = ll.estimate_standard_errors_MLE(
            ll.res.x, monte_carlo=True, niter=mle_niter, print_res=False
        )
        ll.prepare_curvature_adjustment()
        adj_post, A = ll.RW_MH(mcmc_niter, adjusted=True, acceptance_lag=acceptance_lag)

        q = np.quantile(post, quantiles, axis=0).T.flatten()
        q_adj = np.quantile(adj_post, quantiles, axis=0).T.flatten()
        if ll.label == "Debiased Whittle":
            dewhittle_post_quantiles[i] = q
            adj_dewhittle_post_quantiles[i] = q_adj
        else:
            whittle_post_quantiles[i] = q
            adj_whittle_post_quantiles[i] = q_adj

        print(q, q_adj, sep="\n")
        print("")


# stop

import pandas as pd

post_list = ["dewhittle", "adj_dewhittle", "whittle", "adj_whittle"]
param_list = ["rho", "sigma"]
index = pd.MultiIndex.from_product(
    [post_list, param_list, quantiles], names=["posterior", "parameter", "quantile"]
)

new_index = []
for i, idx in enumerate(index):
    *post_params, q = idx
    new_index.append((*post_params, alphas_list[i], q))


k = d * n_q * len(post_list)
posterior_quantiles = np.hstack(
    (
        dewhittle_post_quantiles,
        adj_dewhittle_post_quantiles,
        whittle_post_quantiles,
        adj_whittle_post_quantiles,
    )
).reshape(n_datasets, k)


df = pd.DataFrame(posterior_quantiles, columns=tuple(new_index))
df.columns.names = ["posterior", "parameter", "alpha", "quantile"]


# not a great solution
coverages = {}
for idx in zip(new_index[::2], new_index[1::2]):
    cols = df[list(idx)]
    ll, param, alpha, q = cols.columns[0]
    interval = pd.arrays.IntervalArray.from_arrays(*cols.to_numpy().T, closed="both")
    if "rho" == param:
        count = interval.contains(params[0]).sum()
    else:
        count = interval.contains(params[1]).sum()

    print(f"{ll} coverage for parameter {param} at alpha={alpha}:   {count/n_datasets}")

    coverages[(ll, param, alpha)] = count / n_datasets
    # coverages.append(count/n_datasets)

coverages_arr = np.fromiter(coverages.values(), dtype=float)[None]
df_coverages = pd.DataFrame(coverages_arr, columns=coverages.keys())

plt.plot(alphas, df_coverages["adj_dewhittle", "rho"].to_numpy()[0], ".")
plt.show()
