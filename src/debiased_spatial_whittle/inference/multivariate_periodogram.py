from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.caching import Freezable, ban_if_frozen, lru_cache_frozen
from debiased_spatial_whittle.inference.tapers import Taper, ConstantTaper
from debiased_spatial_whittle.sampling.samples import SampleOnRectangularGrid

xp = BackendManager.get_backend()


fftn, ifftn = BackendManager.get_fft_methods()
ones = BackendManager.get_ones()

from typing import List


class Periodogram(Freezable):
    """
    This class defines a periodogram for a multivariate random field.
    """

    def __init__(self, taper: Taper = None):
        self.fold = True
        self.taper = taper if taper is not None else ConstantTaper()
        super().__init__()
        self.freeze()

    @property
    def taper(self) -> Taper:
        return self._taper

    @taper.setter
    @ban_if_frozen
    def taper(self, value: Taper):
        self._taper = value

    @lru_cache_frozen
    def __call__(self, sample: SampleOnRectangularGrid, return_fft: bool = False) -> xp.ndarray:
        """
        Compute the multivariate periodogram. The data z is expected to be a list
        of p arrays with the same shape, where p is the number of variates.

        Parameters
        ----------
        sample
            Data, list of arrays corresponding to the distinct variates

        return_fft
            If true, returns the Discrete Fourier Transform rather than the periodogram

        Returns
        -------
        periodogram
            Shape (n1, n2, ..., nd, p, p) if the data is p-variate and over d spatial dimensions.
        """
        n_spatial_dims = sample.grid.ndim
        z = sample.values
        j_vec = (
                1
                / xp.sqrt(xp.array(sample.grid.n_points))
                * fftn(z, None, list(range(n_spatial_dims)))
        )
        j_vec = xp.expand_dims(j_vec, -1)
        if return_fft:
            return j_vec
        # first dimensions are spatial dimensions
        j_vec_transpose = xp.conj(xp.transpose(j_vec, (0, 1, -1, -2)))
        p = xp.matmul(j_vec, j_vec_transpose)
        return p
