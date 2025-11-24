from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()

fftn = xp.fft.fftn

from typing import List


class Periodogram:
    """
    This class defines a periodogram for a multivariate random field.
    """

    def __init__(self):
        # TODO allow for tapering in multivariate case
        self.fold = True
        self.taper = lambda shape: xp.ones(shape)

    def __call__(self, z: List[xp.ndarray], return_fft: bool = False) -> xp.ndarray:
        """
        Compute the multivariate periodogram. The data z is expected to be a list
        of p arrays with the same shape, where p is the number of variates.

        Parameters
        ----------
        z
            Data, list of arrays corresponding to the distinct variates

        return_fft
            If true, returns the Discrete Fourier Transform rather than the periodogram

        Returns
        -------
        periodogram
            Shape (n1, n2, ..., nd, p, p) if the data is p-variate and over d spatial dimensions.
        """
        n_spatial_dims = z[0].ndim
        z = xp.stack(z, axis=-1)
        # TODO generalize to 3d
        j_vec = (
                1
                / xp.sqrt(xp.array(z.shape[0] * z.shape[1]))
                * fftn(z, None, list(range(n_spatial_dims)))
        )
        j_vec = xp.expand_dims(j_vec, -1)
        if return_fft:
            return j_vec
        # first dimensions are spatial dimensions
        if BackendManager.backend_name in ("numpy", "cupy"):
            j_vec_transpose = xp.conj(xp.transpose(j_vec, (0, 1, -1, -2)))
        elif BackendManager.backend_name == "torch":
            j_vec_transpose = xp.conj(xp.transpose(j_vec, -1, -2))
        p = xp.matmul(j_vec, j_vec_transpose)
        return p
