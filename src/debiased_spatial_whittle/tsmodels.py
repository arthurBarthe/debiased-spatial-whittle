import numpy as np

from debiased_spatial_whittle.models import (
    CovarianceModel,
    CompoundModel,
    ModelParameter,
)


class SeparableSpaceTimeModel(CompoundModel):
    """
    Class that allows the user to define a separable space-time model by providing the models for the spatial
    and temporal components.
    By convention, the temporal dimension is set to be the last one.
    """

    def __init__(
        self, spatial_model, temporal_model, n_spatial_dims: int = 1, *args, **kwargs
    ):
        super().__init__(children=[spatial_model, temporal_model])
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.n_spatial_dims = n_spatial_dims

    def split_dims(self, lags):
        return lags[: self.n_spatial_dims, ...], lags[self.n_spatial_dims :, ...]

    def cross_product(self, acv_spatial, acv_temporal):
        acv_spatial = np.expand_dims(acv_spatial, -1)
        return acv_spatial * acv_temporal

    def _compute(self, lags: np.ndarray):
        spatial_lags, time_lags = self.split_dims(lags)
        acv_spatial = self.spatial_model(spatial_lags)
        acv_time = self.temporal_model(time_lags)
        return self.cross_product(acv_spatial, acv_time)


class FractionalGaussianNoise(CovarianceModel):
    hurst = ModelParameter(default=0.75, bounds=(0.5, 1))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute(self, lags: np.ndarray):
        lags = lags[0, :]
        two_h = 2 * self.hurst
        return (
            1
            / 2
            * (
                np.abs(lags + 1) ** two_h
                + np.abs(lags - 1) ** two_h
                - 2 * np.abs(lags) ** two_h
            )
        )
