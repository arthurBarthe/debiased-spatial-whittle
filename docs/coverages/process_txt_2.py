import re
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import SquaredExponentialModel

mpl.rcParams.update({"font.size": 16})
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False


def get_file_name(n: int, C: str = "C5") -> str:
    """Get name of file (square grids)."""
    print(n)
    return f"{C}/{C}_DeWhittle_{n}x{n}_SquaredExponentialModel.txt"


C = "C6"
ns = (64, 128, 256, 512)
# ns = [(n,)*2 for n in (64, 128, 256, 512)]
d = {n: np.loadtxt(get_file_name(n, C=C), skiprows=2, max_rows=500) for n in ns}


rho, sigma, nugget = 7.0, 3.0, 0.1  # pick smaller rho

prior_mean = np.array([rho, sigma])
prior_cov = np.array([[1.0, 0.0], [0.0, 0.1]])

# grid = RectangularGrid(n)
model = SquaredExponentialModel()  # TODO: try sq. exponential model!!!

prior_label = rf"$\rho \sim N({prior_mean[0]}, {np.diag(prior_cov)[0]})$, $\sigma \sim N({prior_mean[1]}, {np.diag(prior_cov)[1]})$ "

from scipy import stats

unif = stats.uniform(0, 1)
qs = np.linspace(0, 1, 100)
theory_quants = unif.ppf(qs)

cm1 = plt.get_cmap("Blues")
cm2 = plt.get_cmap("Greens")
fig, ax = plt.subplots(2, 1, figsize=(15, 15))
fig.suptitle(f"QQ plots, {C} adjustment, {model.name}, {prior_label}")
ax[0].plot(
    theory_quants, theory_quants, c="r", linewidth=3, label="std. uniform", zorder=10
)
for i, n in enumerate(ns):
    p = d[n][:, -4:-2]
    adj_p = d[n][:, -2:]
    ax[0].plot(
        theory_quants,
        np.quantile(p[:, 0], qs),
        c=cm1(i / len(ns) + 0.2),
        linewidth=3,
        label=f"{n=}",
        zorder=i,
    )
    ax[0].plot(
        theory_quants,
        np.quantile(adj_p[:, 0], qs),
        c=cm2(i / len(ns) + 0.2),
        linewidth=3,
        label=f"{n=}",
        zorder=i,
    )

ax[0].legend(fontsize=18)

ax[1].plot(
    theory_quants, theory_quants, c="r", linewidth=3, label="std uniform", zorder=10
)
for i, n in enumerate(ns):
    p = d[n][:, -4:-2]
    adj_p = d[n][:, -2:]
    ax[1].plot(
        theory_quants,
        np.quantile(p[:, 1], qs),
        c=cm1(i / len(ns) + 0.1),
        linewidth=3.0,
        label=f"{n=}",
    )
    ax[1].plot(
        theory_quants,
        np.quantile(adj_p[:, 1], qs),
        c=cm2(i / len(ns) + 0.2),
        linewidth=3.0,
        label=f"{n=}",
    )
# ax[1].legend()

ax[0].set_xlabel(r"$\rho$", fontsize=26)
ax[1].set_xlabel(r"$\sigma$", fontsize=26)
for i in range(2):
    ax[i].set_xlim([0, 1])
    ax[i].set_ylim([-0.01, 1.01])

fig.tight_layout()
plt.show()


#### coverages ####


def get_coverages(arr: np.ndarray) -> None:
    """
    Prints the coverages of the posteriors from
    the output of bayesian_coverages.py.
    """
    n = len(arr)
    quantiles = [0.025, 0.975, 0.4, 0.6, 0.5]
    nq = len(quantiles)
    idxs = [2, 7, 12, 17, 22]
    params, q_rhos, q_sigmas, q_adj_rhos, q_adj_sigmas, _ = np.split(arr, idxs, axis=1)

    params = {"rhos": params[:, 0], "sigmas": params[:, 1]}
    q_rhos = {q: q_rho for q, q_rho in zip(quantiles, q_rhos.T)}
    adj_q_rhos = {q: q_rho for q, q_rho in zip(quantiles, q_adj_rhos.T)}

    q_sigmas = {q: q_adj_rhos for q, q_adj_rhos in zip(quantiles, q_sigmas.T)}
    adj_q_sigmas = {
        q: q_adj_sigmas for q, q_adj_sigmas in zip(quantiles, q_adj_sigmas.T)
    }

    rhos = params["rhos"]
    sigmas = params["sigmas"]

    dewhittle = {}
    adj_dewhittle = {}
    for q in np.sort(quantiles)[: nq // 2 + 1]:
        # print(q)
        alpha = round((1 - 2 * q) * 100)

        if q == 0.5:
            alpha = "q50"  # this is just a quantile
            rhos_ = np.sum(q_rhos[q] < rhos)
            adj_rhos = np.sum(adj_q_rhos[q] < rhos)

            sigmas_ = np.sum(q_sigmas[q] < sigmas)
            adj_sigmas = np.sum(adj_q_sigmas[q] < sigmas)

        else:
            rhos_ = np.sum((q_rhos[q] < rhos) & (rhos < q_rhos[1 - q]))
            adj_rhos = np.sum((adj_q_rhos[q] < rhos) & (rhos < adj_q_rhos[1 - q]))

            sigmas_ = np.sum((q_sigmas[q] < sigmas) & (sigmas < q_sigmas[1 - q]))
            adj_sigmas = np.sum(
                (adj_q_sigmas[q] < sigmas) & (sigmas < adj_q_sigmas[1 - q])
            )

        dewhittle[alpha] = {"rho": round(rhos_ / n, 2), "sigma": round(sigmas_ / n, 2)}
        adj_dewhittle[alpha] = {
            "rho": round(adj_rhos / n, 2),
            "sigma": round(adj_sigmas / n, 2),
        }

    label = ["DeWhittle", "adjusted_DeWhittle"]
    for i, dic in enumerate([dewhittle, adj_dewhittle]):
        print(label[i])
        for key, value in dic.items():
            print(f"{key}% {value}")

    return


for n, arr in d.items():
    print(n)
    get_coverages(arr)
    print()
