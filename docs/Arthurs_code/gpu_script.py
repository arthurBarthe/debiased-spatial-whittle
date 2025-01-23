import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
DEVICE = "cuda:0"

torch.manual_seed(1712)

from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("torch")
BackendManager.device = DEVICE

from debiased_spatial_whittle.models import Parameters

import matplotlib.pyplot as plt

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import (
    ExponentialModel,
    SquaredExponentialModel,
    MaternCovarianceModel,
)
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.samples import SampleOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

from typing import Callable
from numpy import ndarray
from numdifftools import Hessian
from scipy.stats import multivariate_normal

from torch.linalg import cholesky, inv, matmul
from debiased_spatial_whittle.models import Parameters


def make_filename(model, prior, grid_size, adjusted):
    prior_info = model.name + str(model.nugget.value)
    prior_info += "__(" + str(prior.mean[0]) + "," + str(prior.cov[0, 0]) + ")"
    prior_info += "__(" + str(prior.mean[1]) + "," + str(prior.cov[1, 1]) + ")"
    return "__".join([prior_info, str(grid_size), str(adjusted)])


def RW_MH(
    niter: int,
    x0: ndarray,
    log_post: Callable,
    prop_cov: ndarray,
    acceptance_lag: int = 500,
    print_res: bool = True,
    approx_grad: bool = False,
    **logpost_kwargs,
):
    """Random walk Metropolis-Hastings: samples the specified posterior"""

    # TODO: mcmc diagnostics
    from time import time

    A = np.zeros(niter, dtype=np.float64)
    U = np.random.rand(niter)

    d = len(prop_cov)
    h = 2.38 / np.sqrt(d)
    props = h * multivariate_normal(np.zeros(d), prop_cov).rvs(size=niter)

    post_draws = np.zeros((niter, d))

    crnt_step = post_draws[0] = x0
    bottom = log_post(crnt_step, **logpost_kwargs)
    # print(bottom)

    if print_res:
        print(f'{f" RW-MH MCMC":-^50}')
    t0 = time()
    for i in range(1, niter):
        prop_step = crnt_step + props[i]
        top = log_post(prop_step, **logpost_kwargs)
        # print(top)

        A[i] = np.min((1.0, np.exp(top - bottom)))
        if U[i] < A[i]:
            crnt_step = prop_step
            bottom = top

        post_draws[i] = crnt_step

        if (i + 1) % acceptance_lag == 0:
            print(
                f"Iteration: {i+1}    Acceptance rate: {A[i-(acceptance_lag-1): (i+1)].mean().round(3)}    Time: {np.round(time()-t0,3)}s"
            )

    return post_draws


model = SquaredExponentialModel()
model.nugget = 0.05

est_params = Parameters((model.rho, model.sigma))
model.rho = 14
model.sigma = 2

shape = (1000, 1000)
mask_france = grids.ImgGrid(shape).get_new()
grid_france = RectangularGrid(shape)
grid_france.mask = torch.ones(shape, device=DEVICE)
# grid_france.mask = mask_france
sampler = SamplerOnRectangularGrid(model, grid_france)
sampler.n_sims = 50

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_france, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)
options = dict(
    ftol=np.finfo(float).eps * 10, gtol=1e-20
)  # for extreme precision, ftol is stops iteration, gtol is for gradient
estimator = Estimator(debiased_whittle, use_gradients=False, optim_options=options)


prior = multivariate_normal([12.0, 1.2], np.array([[1.5, 0.0], [0.0, 0.25]]))
ADJUSTED = True

# number of independent samples
n_samples = 250
# number of mcmc steps
n_mcmc = 2000

filename = make_filename(model, prior, grid_france.n, ADJUSTED)
print(filename)

try:
    data = np.loadtxt(filename)
    qs = list(data)
except:
    qs = []


for i_sample in range(n_samples):
    # sample parameters from prior
    theta0 = prior.rvs()
    model.rho, model.sigma = theta0
    print(f"{i_sample=}")
    print(f"theta0={theta0.round(3)}")
    # sample random field
    sampler = SamplerOnRectangularGrid(model, grid_france)
    z = sampler()
    # compute dbw at true parameter for later check
    dbw_0 = debiased_whittle(z, model)
    # point estimate
    model.rho, model.sigma = None, None
    estimate = estimator(model, z)
    dbw_hat = debiased_whittle(z, estimate)
    if dbw_0 < dbw_hat:
        print("Optimization failed")
        continue

    rho_hat, sigma_hat = estimate.rho.value, estimate.sigma.value
    theta_hat = torch.tensor([rho_hat, sigma_hat]).reshape((-1, 1))
    MAP = torch.tensor([rho_hat, sigma_hat])
    print(f"MAP={MAP.numpy().round(3)}")
    # Compute adjustment matrix
    params_for_gradient = Parameters((estimate.rho, estimate.sigma))
    H = debiased_whittle.fisher(estimate, params_for_gradient)
    Jhat = debiased_whittle.jmatrix_sample(
        estimate, params_for_gradient, n_sims=1000, block_size=1
    )
    if ADJUSTED:
        M_A = cholesky(matmul(H, matmul(inv(Jhat), H))).T
        M = cholesky(H).T
        C = matmul(inv(M), M_A)
        C = C / np.sqrt(grid_france.n_points / 2)

    # define posterior
    def log_posterior(rho, sigma):
        model.rho, model.sigma = rho, sigma
        return -debiased_whittle(z, model) / 2 * grid_france.n_points + prior.logpdf(
            [rho, sigma]
        )

    def log_posterior_adjusted(rho, sigma):
        u = torch.tensor([rho, sigma]).reshape((-1, 1))
        theta_star = theta_hat + matmul(C.cpu(), u - theta_hat)
        model.rho, model.sigma = theta_star[0].item(), theta_star[1].item()
        return -debiased_whittle(z, model) / 2 * grid_france.n_points + prior.logpdf(
            [rho, sigma]
        )

    if ADJUSTED:
        log_posterior = log_posterior_adjusted

    # define proposal distribution
    h = (
        1
        / 2
        * debiased_whittle.fisher(model, Parameters((model.rho, model.sigma)))
        * grid_france.n_points
    )
    if ADJUSTED:
        # TODO: assert allclose
        h = matmul(C.T, matmul(h, C))
        if not torch.allclose(h, H @ inv(Jhat) @ H):
            continue

    log_post = lambda x: log_posterior(x[0], x[1])
    prop_cov = np.linalg.inv(-Hessian(log_post)(MAP))

    # run mcmc
    posterior_sample = RW_MH(n_mcmc, MAP, log_post, prop_cov)

    q = np.mean(theta0 <= posterior_sample[n_mcmc // 2 :,], axis=0)
    print(f"{q=}")
    qs.append(q)
    with open(filename, "a") as f:
        np.savetxt(f, q.reshape((1, -1)))
    # plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(posterior_sample[:, 0])
    ax[0].axhline(theta0[0], c="r")
    ax[1].plot(posterior_sample[:, 1])
    ax[1].axhline(theta0[1], c="r")
    quantiles = np.quantile(np.array(qs), np.linspace(0, 1, 100), axis=0)
    ax[2].plot(np.linspace(0, 1, 100), quantiles)
    ax[2].plot([0, 1], [0, 1])
    ax[2].legend(("rho", "sigma"), fontsize=16)
    fig.tight_layout()
    plt.show()
