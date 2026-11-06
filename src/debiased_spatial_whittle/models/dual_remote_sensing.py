import numpy as np
from debiased_spatial_whittle.models.base import CovarianceModel, CompoundModel, ModelParameter
from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()
inv = BackendManager.get_inv()


class DualRemoteSensing(CompoundModel):
    """
    Class for the covariance model corresponding to a univariate field measured via two channels.
    
    This model represents the scenario where a single underlying spatial field is observed
    through two different measurement channels, each with its own independent noise.
    
    Attributes
    ----------
    base_model : CovarianceModel
        The covariance model of the underlying latent field
    sigma1 : ModelParameter
        Noise amplitude of channel 1, must be in (0.0, 1.0)
    sigma2 : ModelParameter
        Noise amplitude of channel 2, must be in (0.0, 1.0)
    
    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
    >>> latent_model = SquaredExponentialModel(rho=5.0)
    >>> model = DualRemoteSensing(latent_model, sigma1=0.1, sigma2=0.2)
    """

    sigma1 = ModelParameter(
        default=0.1, bounds=(0.0, 1.0), doc="Noise amplitude of channel 1"
    )
    sigma2 = ModelParameter(
        default=0.1, bounds=(0.0, 1.0), doc="Noise amplitude of channel 2"
    )

    def __init__(self, base_model: CovarianceModel, *args, **kwargs):
        super().__init__(
            children=[
                base_model,
            ],
            *args,
            **kwargs,
        )

    @property
    def base_model(self):
        return self.children[0]

    def compute11(self, lags: np.ndarray):
        acv_latent = self.children[0]._compute(lags)
        return acv_latent + self.sigma1 * np.all(lags == 0, 0)

    def compute22(self, lags: np.ndarray):
        acv_latent = self.children[0]._compute(lags)
        return acv_latent + self.sigma2 * np.all(lags == 0, 0)

    def _compute(self, lags: np.ndarray):
        acv_latent = self.children[0]._compute(lags)
        col1 = np.stack(
            (acv_latent + self.sigma1 * np.all(lags == 0, 0), acv_latent), -1
        )
        col2 = np.stack(
            (acv_latent, acv_latent + self.sigma2 * np.all(lags == 0, 0)), -1
        )
        return np.stack((col1, col2), -1)

    def _gradient(self, lags: np.ndarray):
        raise NotImplementedError()

    def cov_mat_x1_x2(self, x1: np.ndarray, x2: np.ndarray = None) -> np.ndarray:
        """
        For multivariate data, if x1 has n1 locations and x2 has n2 locations, we return a matrix
        with size (n1 * 2, n1 * 2)
        Parameters
        ----------
        x1
        x2

        Returns
        -------

        """
        raise NotImplementedError()
        if x2 is None:
            x2 = x1
        x1 = np.expand_dims(x1, axis=1)
        x2 = np.expand_dims(x2, axis=0)
        lags = x1 - x2
        lags = np.transpose(lags, (2, 0, 1))
        acv = self(lags)
        acv = np.transpose(acv, (3, 1, 4, 2))
        return np.reshape(acv, (2 * x1.shape[0], 2 * x2.shape[0]))

    def predict(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        x_pred: np.ndarray,
        return_variance: bool = False,
        full_model: CovarianceModel = None,
    ):
        """
        Compute conditional mean at a set of locations x_pred given values y_obs observed at x_obs.

        Parameters
        ----------
        x_obs
            shape (n_obs, d), array of locations where observations are made
        y_obs
            shape (n_obs, 1), observed values
        x_pred
            shape (n_pred, d), array of locations where predicted values are requested
        return_variance
            If true, return the covariance matrix. Not implemented yet.
        full_model
            Covariance model of the data. By default, same as the current model. This can be used to separate
            several additive components.

        Returns
        -------
        y_pred
            shape (n_pred, 1), array of predicted values
        """
        x_obs1, x_obs2 = x_obs
        y_obs1, y_obs2 = y_obs

        # first component
        x_obs1 = np.expand_dims(x_obs1, 1)
        # x_obs (n_obs, 1, d)
        lags_x1x1 = x_obs1 - np.transpose(x_obs1, (1, 0, 2))
        # lags_xx (n_obs, n_obs, d)

        cov_mat_x1x1 = self.compute11(np.transpose(lags_x1x1, (2, 0, 1)))

        # second component
        x_obs2 = np.expand_dims(x_obs2, 1)
        # x_obs (n_obs, 1, d)
        lags_x2x2 = x_obs2 - np.transpose(x_obs2, (1, 0, 2))
        # lags_xx (n_obs, n_obs, d)

        cov_mat_x2x2 = self.compute22(np.transpose(lags_x2x2, (2, 0, 1)))

        # cross
        lags_x1x2 = x_obs1 - np.transpose(x_obs2, (1, 0, 2))
        cov_mat_x1x2 = self.base_model(np.transpose(lags_x1x2, (2, 0, 1)))

        cov_mat_xx = np.concatenate(
            (
                np.concatenate((cov_mat_x1x1, cov_mat_x1x2), 1),
                np.concatenate((cov_mat_x1x2.T, cov_mat_x2x2), 1),
            ),
            0,
        )
        # cov_mat_xx (n_obs, n_obs)
        cov_mat_xx_inv = inv(cov_mat_xx)

        x_pred = np.expand_dims(x_pred, 1)
        # x_pred (n_pred, 1, d)

        lags_xpx1 = x_pred - np.transpose(x_obs1, (1, 0, 2))
        # lags_yx (n_pred, n_obs, d)
        sigma_xpx1 = self.base_model(np.transpose(lags_xpx1, (2, 0, 1)))

        lags_xpx2 = x_pred - np.transpose(x_obs2, (1, 0, 2))
        # lags_yx (n_pred, n_obs, d)
        sigma_xpx2 = self.base_model(np.transpose(lags_xpx2, (2, 0, 1)))

        sigma_yx = np.concatenate((sigma_xpx1, sigma_xpx2), 1)
        # sigma_yx (n_pred, n_obs)

        weights = np.dot(sigma_yx, cov_mat_xx_inv)
        # weights (n_pred, n_obs)
        y_pred = np.matmul(weights, np.concatenate((y_obs1, y_obs2), 0))
        return y_pred