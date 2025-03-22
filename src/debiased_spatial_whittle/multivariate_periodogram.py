from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

fftn = np.fft.fftn

from typing import List


class Periodogram:
    """
    This class defines a periodogram for a multivariate random field.
    """

    def __init__(self):
        # TODO allow for tapering in multivariate case
        self.fold = True
        self.taper = lambda shape: np.ones(shape)

    def __call__(self, z: List[np.ndarray], return_fft: bool = False) -> np.ndarray:
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
        z = np.stack(z, axis=-1)
        # TODO generalize to 3d
        j_vec = (
            1
            / np.sqrt(np.array(z.shape[0] * z.shape[1]))
            * fftn(z, None, list(range(n_spatial_dims)))
        )
        j_vec = np.expand_dims(j_vec, -1)
        if return_fft:
            return j_vec
        # first dimensions are spatial dimensions
        if BackendManager.backend_name in ("numpy", "cupy"):
            j_vec_transpose = np.conj(np.transpose(j_vec, (0, 1, -1, -2)))
        elif BackendManager.backend_name == "torch":
            j_vec_transpose = np.conj(np.transpose(j_vec, -1, -2))
        p = np.matmul(j_vec, j_vec_transpose)
        return p
