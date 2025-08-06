import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fftshift

from debiased_spatial_whittle.grids import RectangularGrid


def prod_list(l):
    if len(l) == 0:
        return 1
    else:
        return l[0] * prod_list(l[1:])


def plot_fourier_values(
    grid: RectangularGrid,
    values: np.ndarray,
    plot_func: str = "imshow",
    ax=None,
    *args,
    **kwargs,
):
    """
    Helper function to plot spectral-domain values (e.g. a periodogram, or spectral residuals) with
    the correct axes.

    Parameters
    ----------
    grid
        Spatial grid of the data
    values
        Spectral domain values used for the plot
    plot_func
        matplotlib function used for the plot. Can be either imshow, pcolor, contour or contourf

    Returns
    -------
    axis
        pyplot axis
    """
    frequencies = grid.fourier_frequencies
    frequencies = fftshift(frequencies, (0, 1))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    if plot_func == "imshow":
        extent = (
            frequencies[0, 0, 0],
            frequencies[-1, 0, 0],
            frequencies[0, 0, 1],
            frequencies[0, -1, 1],
        )
        im = ax.imshow(values, extent=extent, *args, **kwargs)
        plt.colorbar(im)
    return ax


def video_plot_3d(
    y: np.ndarray,
    interval=100,
    repeat_delay=1000,
    get_title=None,
    xlabel="",
    ylabel="",
    **imshow_kwargs,
):
    """
    Produces an animated plot to show the 3 dimensional array
    Parameters
    ----------
    y
        3d data to be plotted.
    """

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(y.shape[-1]):
        im = ax.imshow(y[..., i], animated=True, **imshow_kwargs)
        if get_title is not None:
            ttl = plt.text(
                0.0,
                1.01,
                get_title(i),
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax.transAxes,
            )
        if i == 0:
            ax.imshow(y[..., 0], **imshow_kwargs)  # show an initial one first
        ims.append([im, ttl])
    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, blit=True, repeat_delay=repeat_delay
    )

    return ani
