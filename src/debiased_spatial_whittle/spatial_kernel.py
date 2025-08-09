from typing import Tuple
from debiased_spatial_whittle.backend import BackendManager


xp = BackendManager.get_backend()
fftn = xp.fft.fftn
ifftn = xp.fft.ifftn


def spatial_kernel(
    g: xp.ndarray, m: Tuple[int, int] = (0, 0), n_spatial_dim: int = None
) -> xp.ndarray:
    r"""
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

    Examples
    --------
    >>> g = xp.arange(10)
    >>> spatial_kernel(g)
    array([2.85000000e+01, 2.40000000e+01, 1.96000000e+01, 1.54000000e+01,
           1.15000000e+01, 8.00000000e+00, 5.00000000e+00, 2.60000000e+00,
           9.00000000e-01, 2.09423122e-15, 2.09423122e-15, 9.00000000e-01,
           2.60000000e+00, 5.00000000e+00, 8.00000000e+00, 1.15000000e+01,
           1.54000000e+01, 1.96000000e+01, 2.40000000e+01])

    Notes
    -----
    The formula for the spatial kernel in dimension 1 is,

    $$
        c_g(\tau) = \sum_{s}{g_s g_{s + \tau}}, \quad \tau=0, \ldots, n - 1, - (n - 1), \ldots, -1.
    $$
    """
    if n_spatial_dim is None:
        n_spatial_dim = g.ndim
    n = g.shape[:n_spatial_dim]
    normalization_factor = xp.prod(xp.array(n))
    two_n = tuple([s * 2 - 1 for s in n])
    if m == (0, 0):
        if n_spatial_dim == g.ndim:
            # univariate case
            f = xp.abs(fftn(g, two_n)) ** 2
            cg = ifftn(f)
            cg /= normalization_factor
            return xp.real(cg)
        else:
            # multivariate case
            g = xp.expand_dims(g, -1)
            f1 = fftn(g, two_n, axes=tuple(range(n_spatial_dim)))
            f2 = xp.transpose(f1, tuple(range(n_spatial_dim)) + (-1, -2))
            cg = ifftn(xp.matmul(f1, f2.conj()), axes=tuple(range(n_spatial_dim)))
            cg /= normalization_factor
            return xp.real(cg)
    # TODO this specific case only works in 2d right now
    m1, m2 = m
    n1, n2 = n
    a = xp.exp(2j * xp.pi * m1 / n1 * xp.arange(n1)).reshape((-1, 1))
    a = a * xp.exp(2j * xp.pi * m2 / n2 * xp.arange(n2)).reshape((1, -1))
    g2 = g * a
    f = fftn(g, two_n) * xp.conj(fftn(g2, two_n))
    cg = ifftn(f)
    # TODO check normalization is consistent
    cg /= xp.sum(g**2)
    return cg
