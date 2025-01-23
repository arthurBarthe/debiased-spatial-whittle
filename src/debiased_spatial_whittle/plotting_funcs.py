from math import prod
import numpy as np
from numpy import ndarray

try:
    import seaborn as sns
except:
    pass
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from typing import Union, List


# TODO: make 1d work
def plot_marginals(
    list_draws: List[ndarray],
    truths: Union[ndarray, None] = None,
    title: Union[str, None] = None,
    axis_labels: Union[List, None] = None,
    density_labels: Union[List, None] = None,
    shape: Union[tuple, None] = None,
    figsize: tuple = (15, 7),
    cmap: Union[str, None] = None,
    **plotargs,
):
    """draws: list of arrays of samples to plot"""
    # TODO: fix single 1-d case, make some tests!
    dims = [draws.shape[1] if draws.ndim > 1 else 1 for draws in list_draws]
    nplots = max(dims)
    if nplots == 1:
        nplots = sum(dims)

    ndistributions = len(list_draws)
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if axis_labels is None:
        axis_labels = list(range(nplots))

    if shape is None:
        shape = (1, nplots)
    else:
        assert prod(shape) >= nplots, "shape not >= dim(theta)"

    if cmap is None:
        cmap = "viridis"
    cm = plt.get_cmap(cmap)

    fig, ax_list = plt.subplots(*shape, figsize=figsize)
    for i, ax in enumerate(ax_list.flatten()):
        if i >= nplots:
            fig.delaxes(ax_list.flatten()[i])
            continue

        if truths is not None:
            ax.axvline(
                truths[i],
                c="k",
                linewidth=3.0,
                linestyle="--",
                zorder=3,
                label="True value",
            )

        if max(dims) == 1:  # for 1-d case
            sns.kdeplot(
                list_draws[i],
                fill=False,
                ax=ax,
                legend=False,
                label="density",
                linewidth=5.0,
                color=cm(1 / ndistributions),
                **plotargs,
            )
        else:
            for j in range(ndistributions):
                if i > dims[j] - 1:
                    continue
                sns.kdeplot(
                    list_draws[j][:, i],
                    fill=False,
                    ax=ax,
                    legend=False,
                    linewidth=3.0,
                    color=cm(j / ndistributions),
                    **plotargs,
                )

        ax.set_xlabel(axis_labels[i], fontsize=24)
        ax.tick_params(axis="x", labelsize=18)

        ax.set_yticks([])
        ax.set_ylabel("")  # .axes.get_yaxis().set_visible(False)
        ax.spines[["top", "left", "right"]].set_visible(False)

    if title is None:
        title = "Marginal densities"
    fig.suptitle(title, fontsize=26, y=1.10)

    if density_labels is None:
        if max(dims) == 1:
            legend_labels = ["density"]
        else:
            legend_labels = [f"density #{i}" for i in range(ndistributions)]
    else:
        legend_labels = density_labels.copy()
    if truths is not None:
        legend_labels.insert(0, "True value")

    # ax_list[1].set_xlim([-1,1])     # bounds
    # if max(dims)==1:
    # print(ax.get_legend_handles_labels())
    # fig.legend(*ax.get_legend_handles_labels(), fontsize=20)
    fig.legend(legend_labels, fontsize=20, bbox_to_anchor=(1.10, 1.12))
    fig.tight_layout()
    plt.show()
    return


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
