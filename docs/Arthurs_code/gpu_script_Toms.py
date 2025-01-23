from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("jax")
np = BackendManager.get_backend()
# BackendManager.device = DEVICE


import matplotlib.pyplot as plt

plt.rcParams.update(
    {"font.size": 18.0, "axes.spines.top": False, "axes.spines.right": False}
)

from debiased_spatial_whittle.bayes.funcs import (
    transform,
    RW_MH,
    compute_hessian,
    svd_decomp,
)
from debiased_spatial_whittle.likelihood import DebiasedWhittle
import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    MaternCovarianceModel,
)
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid

# from debiased_spatial_whittle.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.bayes import (
    DeWhittle,
    Whittle,
    Gaussian,
    MCMC,
    GaussianPrior,
    Optimizer,
)

import numpy
from typing import Callable, Tuple
from scipy.stats import multivariate_normal


def make_filename(model, prior, grid_size: Tuple, adjusted: bool, C: str):
    prior_info = model.name + str(model.nugget.value)
    prior_info += "__(" + str(prior.mean[0]) + "," + str(prior.cov[0, 0]) + ")"
    prior_info += "__(" + str(prior.mean[1]) + "," + str(prior.cov[1, 1]) + ")"
    return "__".join([prior_info, str(grid_size), str(adjusted), C])


n = (1000, 1000)
grid = RectangularGrid(n)
# mask_france = grids.ImgGrid(shape).get_new()
# grid_france = RectangularGrid(shape)
# grid_france.mask = np.ones(shape)
# grid_france.mask = mask_france


prior = multivariate_normal([12.0, 1.2], np.array([[0.5, 0.0], [0.0, 0.0625]]))
_prior = GaussianPrior(prior.mean, prior.cov)  # make sure sigma not negative

# number of independent data-sets  # TODO: change variable name
n_datasets = 250
# number of mle samples
n_sims = 500
# number of mcmc steps
n_mcmc = 4000


nugget = 0.05
model = SquaredExponentialModel()
model.nugget = nugget

filename_unadj = make_filename(model, prior, grid.n, False, "None")
filename_adj_C2 = make_filename(model, prior, grid.n, True, "C2")
filename_adj_C4 = make_filename(model, prior, grid.n, True, "C4")
filename_adj_C5 = make_filename(model, prior, grid.n, True, "C5")

files = [filename_unadj, filename_adj_C2, filename_adj_C4, filename_adj_C5]
print(filename_unadj)

# stop

qs = []
for filename in files:
    try:
        data = numpy.loadtxt(filename)
        qs.append(list(data))
    except:
        qs.append([])


for i_dataset in range(n_datasets):
    # sample parameters from prior
    theta0 = prior.rvs()
    model.rho, model.sigma = theta0
    model.nugget = nugget
    print(f"Dataset #{i_dataset} \t theta_0 = {theta0.round(3)}")

    # sample random field
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()

    dw = DeWhittle(
        z, grid, SquaredExponentialModel(), nugget=nugget, transform_func=None
    )

    dw.fit(x0=theta0)

    # prepare adjustments
    MLEs = dw.sim_MLEs(dw.res.x, niter=n_sims, print_res=False)

    dw.compute_C4(dw.res.x)
    dw.compute_C5(dw.res.x)

    H = dw.fisher(dw.res.x)
    grads = dw.sim_J_matrix(dw.res.x, niter=n_sims)

    dw.compute_C2()

    # run mcmc
    dw_posterior = MCMC(dw, _prior)

    dw_post_draws = dw_posterior.RW_MH(n_mcmc, adjusted=False, acceptance_lag=1000)
    adj2_dw_post = dw_posterior.RW_MH(
        n_mcmc, adjusted=True, acceptance_lag=1000, C=dw.C2
    )
    adj4_dw_post = dw_posterior.RW_MH(
        n_mcmc, adjusted=True, acceptance_lag=1000, C=dw.C4
    )
    adj5_dw_post = dw_posterior.RW_MH(
        n_mcmc, adjusted=True, acceptance_lag=1000, C=dw.C5
    )

    posts = [dw_post_draws, adj2_dw_post, adj4_dw_post, adj5_dw_post]

    # save results to file
    for i, (filename, post) in enumerate(zip(files, posts)):
        q = np.mean(theta0 <= post[n_mcmc // 2 :,], axis=0)
        qs[i].append(q)
        print(f"q={q.round(2)}")
        with open(filename, "a") as f:
            numpy.savetxt(f, q.reshape((1, -1)))

    # plot
    fig, ax = plt.subplots(len(qs), 3, figsize=(15, 20))
    for i in range(len(qs)):
        for j in range(3):
            if j < 2:
                ax[i, j].axhline(theta0[j], c="r", linewidth=3, zorder=10)
                ax[i, j].plot(posts[i][:, j])

            else:
                quantiles = np.quantile(np.array(qs[i]), np.linspace(0, 1, 100), axis=0)
                ax[i, j].plot(np.linspace(0, 1, 100), quantiles, linewidth=3)
                ax[i, j].plot([0, 1], [0, 1], linewidth=3, c="k", zorder=0)
                ax[i, j].legend(("rho", "sigma"), fontsize=22)

    # fig.legend()
    fig.tight_layout()
    plt.show()
