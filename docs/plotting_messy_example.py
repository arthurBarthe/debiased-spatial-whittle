from scipy.stats import gaussian_kde
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150, figsize=(15, 7)):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0 1]")
    curves = []
    ys = []

    # xx = np.linspace(np.min(np.concatenate(data)),
    # np.max(np.concatenate(data)), n_points)
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i * (1.0 - overlap)
        ys.append(y)

        xx = np.linspace(np.min(data[i]), np.max(data[i]), n_points)
        density = pdf(xx)
        if fill:
            plt.fill_between(
                xx,
                np.ones(n_points) * y,
                density + y,
                zorder=len(data) - i + 1,
                color=fill,
            )
        plt.plot(xx, density + y, c="k", zorder=len(data) - i + 1, lw=1.0)
    if labels:
        plt.yticks(ys, labels, fontsize=14)


plt.figure(figsize=(15, 7))
plt.title(
    "deWhittle MLEs $t$-random field, Sq.Exp. kernel, n=(64,64)",
    loc="center",
    fontsize=20,
    color="gray",
)
dists = [mles[:, 0] for mles in MLEs]
ridgeline(dists[::-1], overlap=0.0, fill="red", labels=dfs[::-1])
dists = [mles[:, 1] for mles in MLEs]
ridgeline(dists[::-1], overlap=0.0, fill="y", labels=dfs[::-1])

plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().yaxis.grid(True)
plt.ylabel(r"$\nu$", fontsize=18)
# plt.xlabel(r'$x$', fontsize=18)
plt.axvline(params[0], zorder=999, c="k")
plt.axvline(params[1], zorder=999, c="k")
plt.xticks(fontsize=14)
plt.xlim([-1, 3])

rho_patch = mpatches.Patch(color="red", label=r"log$\rho$")
sigma_patch = mpatches.Patch(color="y", label=r"log$\sigma$")
plt.legend(handles=[rho_patch, sigma_patch], fontsize=16)

plt.tight_layout()
plt.show()
