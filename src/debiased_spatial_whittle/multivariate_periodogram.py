import numpy as np
from numpy.fft import fftn, ifftn
from typing import List

from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import ExpectedPeriodogram
from debiased_spatial_whittle.models import BivariateUniformCorrelation, TransformedModel


class Periodogram:
    """
    This class defines a periodogram for a multivariate random field.
    """
    def __init__(self):
        pass

    def __call__(self, z: List[np.ndarray], return_fft: bool = False):
        n_spatial_dims = z[0].ndim
        z = np.stack(z, axis=-1)
        j_vec = 1 / np.sqrt(z.shape[0] * z.shape[1]) * fftn(z, axes=list(range(n_spatial_dims)))
        j_vec = np.expand_dims(j_vec, -1)
        if return_fft:
            return j_vec
        # first dimensions are spatial dimensions
        j_vec_transpose = np.conj(np.transpose(j_vec, (0, 1, -1, -2)))
        p = np.matmul(j_vec, j_vec_transpose)
        return p


class TransformedExpectedPeriodogram:
    def __init__(self, grid: RectangularGrid, periodogram: Periodogram, ep: ExpectedPeriodogram):
        self.grid = grid
        self.periodogram = periodogram
        self.e_periodogram = ep

    def __call__(self, model: TransformedModel):
        input_model = model.input_model
        input_ep = self.e_periodogram(input_model)
        transform = model.transform_on_grid(self.grid.fourier_frequencies)
        transform_transpose = np.transpose(transform, (0, 1, -1, -2))
        term1 = np.matmul(input_ep, transform_transpose)
        return np.matmul(transform, term1)

