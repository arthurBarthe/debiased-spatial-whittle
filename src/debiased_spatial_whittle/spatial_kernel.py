from .backend import BackendManager

np = BackendManager.get_backend()
from typing import Tuple

fftn = np.fft.fftn
ifftn = np.fft.ifftn


def spatial_kernel(
    g: np.ndarray, m: Tuple[int, int] = (0, 0), n_spatial_dim: int = None
) -> np.ndarray:
    """
    Compute the spatial kernel, cg in the paper, via FFT for computational efficiency.

    Parameters
    ----------
    g
        mask of observations, or more generally pointwise modulation e.g. a taper or the product of a taper with an
        observation mask.
        Shape (n1, ..., nd) for univariate data in d-dimensional space
        Shape (n1, ..., nd, p) for p-variate data in d-dimensional space

    n_spatial_dim
        Number of dimensions that are spatial dimensions. In the multivariate case, the last dimension is used for the
        different variates.

    m
        offset in frequency indices

    Returns
    -------
    cg
        Spatial kernel.
        Shape (2 * n1 + 1, ..., 2 * nd + 1) for univariate data
        Shape (2 * n1 + 1, ..., 2 * nd + 1, p, p) for p-variate data
    """
    if n_spatial_dim is None:
        n_spatial_dim = g.ndim
    n = g.shape[:n_spatial_dim]
    normalization_factor = np.prod(np.array(n))
    two_n = tuple([s * 2 - 1 for s in n])
    if m == (0, 0):
        if n_spatial_dim == g.ndim:
            # univariate case
            f = np.abs(fftn(g, two_n)) ** 2
            cg = ifftn(f)
            cg /= normalization_factor
            return np.real(cg)
        else:
            # multivariate case
            g = np.expand_dims(g, -1)
            f1 = fftn(g, two_n, axes=tuple(range(n_spatial_dim)))
            f2 = np.transpose(f1, tuple(range(n_spatial_dim)) + (-1, -2))
            cg = ifftn(np.matmul(f1, f2.conj()), axes=tuple(range(n_spatial_dim)))
            cg /= normalization_factor
            return np.real(cg)
    # TODO this specific case only works in 2d right now
    m1, m2 = m
    n1, n2 = n
    a = np.exp(2j * np.pi * m1 / n1 * np.arange(n1)).reshape((-1, 1))
    a = a * np.exp(2j * np.pi * m2 / n2 * np.arange(n2)).reshape((1, -1))
    g2 = g * a
    f = fftn(g, two_n) * np.conj(fftn(g2, two_n))
    cg = ifftn(f)
    # TODO check normalization is consistent
    cg /= np.sum(g**2)
    return cg
