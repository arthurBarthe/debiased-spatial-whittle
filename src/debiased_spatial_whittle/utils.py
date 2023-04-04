import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def prod_list(l):
    if len(l) == 0:
        return 1
    else:
        return l[0] * prod_list(l[1:])

def video_plot_3d(y: np.ndarray):
    """
    Produces an animated plot to show the 3 dimensional array
    Parameters
    ----------
    y
        3d data to be plotted.
    """

    fig, ax = plt.subplots()

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(y.shape[-1]):
        im = ax.imshow(y[..., i], vmin=-2, vmax=2, cmap='inferno', animated=True)
        if i == 0:
            ax.imshow(y[..., 0], vmin=-2, vmax=2, cmap='inferno')  # show an initial one first
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    plt.show()
    return ani