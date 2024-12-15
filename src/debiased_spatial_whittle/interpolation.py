import autograd.numpy as np
import matplotlib.pyplot as plt
from functools import cached_property

from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.models import CovarianceModel, SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle

# from debiased_spatial_whittle.bayes_old import DeWhittle2
ndarray = np.ndarray
from typing import Optional


class SimpleKriging:
    """Interpolation via simple Kriging"""

    # TODO: approximate for larger grids
    def __init__(self, z: np.ndarray, grid: RectangularGrid, model: CovarianceModel):
        self.z = z  # TODO: change to property
        grid.mask = grid.mask.astype(bool)
        self.grid = grid
        self.model = model

    @cached_property
    def xs_without_missing(self):
        return np.argwhere(self.grid.mask)

    @property
    def n_obs(self):
        return len(self.xs_without_missing)

    @cached_property
    def missing_xs(self):
        return np.argwhere(~self.grid.mask)

    @cached_property
    def lags_without_missing(self):
        return self.compute_lags_without_missing()

    def compute_lags_without_missing(self, closest_idxs: Optional[ndarray] = None):
        """
        Computes the xs on a grid for covariance function:
        closest_idxs are an array of indices closest to some point for approximation
        """
        if closest_idxs is None:
            closest_idxs = np.arange(self.n_obs)

        grid_vec = np.argwhere(self.grid.mask)[None, closest_idxs].T
        lags = grid_vec - np.transpose(
            grid_vec, axes=(0, 2, 1)
        )  # still general for n-dimensions
        return np.array(lags)

    def update_model_params(self, params: ndarray) -> None:
        free_params = self.model.params
        updates = dict(zip(free_params.names, params))
        free_params.update_values(updates)
        return

    def compute_inv_covmat(self, params: ndarray):
        self.update_model_params(params)
        covMat = self.model(self.lags_without_missing)  # Sigma_22
        print(covMat.shape)
        covMat_inv = np.linalg.inv(covMat)
        return covMat_inv

    @staticmethod
    def format_arr(x):
        x = np.array(x)
        if x.ndim == 1:
            xs = x[None].copy()

        elif x.ndim == 2:
            xs = x.copy()
        else:
            raise TypeError("x must be array of point(s), < 2 dim")
        return xs

    def __call__(self, x: ndarray, params: ndarray):
        covMat_inv = self.compute_inv_covmat(params)
        # self.update_model_params(params)

        xs = self.format_arr(x)

        pred_means = np.zeros(len(xs))
        pred_vars = np.zeros(len(xs))
        for i, point in enumerate(xs):
            acf = self.model((self.xs_without_missing - point).T)

            weights = covMat_inv @ acf

            pred_means[i] = np.dot(weights, self.z[self.grid.mask])
            # mean = acf @ covMat_inv @ self.z[self.grid.mask]

            pred_vars[i] = self.model(np.zeros(1)) - np.dot(acf, weights)

        return pred_means, pred_vars

    def bayesian_prediction(self, x: ndarray, posterior_samples: ndarray):
        xs = self.format_arr(x)
        posterior_samples = self.format_arr(posterior_samples)

        m = len(xs)
        ndraws = len(posterior_samples)
        xs_pred_draws = np.zeros((ndraws, m))
        for i, sample in enumerate(posterior_samples):
            print(f"\rComputed {i+1} out of {ndraws} posterior draws", end="")
            pred_means, pred_vars = self(x, np.exp(sample))
            xs_pred = pred_means + np.sqrt(pred_vars) * np.random.randn(m)
            xs_pred_draws[i] = xs_pred
        print()
        return xs_pred_draws

    def find_n_closest(self, x: ndarray, n_closest: int = 1000):
        """finds indices of n closest points on the grid for a specified x"""

        diffs = (self.xs_without_missing - x).T
        d = np.sqrt(sum((lag**2 for lag in diffs)))

        closest_idxs = np.argsort(d)[:n_closest]
        return closest_idxs

    def compute_inv_approx_covMat(
        self, params: ndarray, closest_idxs: ndarray
    ) -> ndarray:
        # TODO: merge this into other inv covMat function
        self.update_model_params(params)

        lags = self.compute_lags_without_missing(closest_idxs)
        approx_covMat = self.model(lags)
        inv_approx_covMat = np.linalg.inv(approx_covMat)
        return inv_approx_covMat

    def approx(self, x: ndarray, params: ndarray, n_closest: int = 1000):
        self.update_model_params(params)

        xs = self.format_arr(x)  # TODO: not using xs

        approx_pred_means = np.zeros(len(xs))
        approx_pred_vars = np.zeros(len(xs))
        for i, point in enumerate(xs):
            closest_idxs = self.find_n_closest(point, n_closest)

            inv_approx_covMat = self.compute_inv_approx_covMat(params, closest_idxs)

            acf = self.model((self.xs_without_missing[closest_idxs] - point).T)

            weights = inv_approx_covMat @ acf

            z_closest = self.z[tuple(self.xs_without_missing[closest_idxs].T)]
            approx_pred_means[i] = np.dot(weights, z_closest)
            approx_pred_vars[i] = self.model(np.zeros(1)) - np.dot(acf, weights)

        return approx_pred_means, approx_pred_vars

    def approx_bayesian_prediction(
        self, x: ndarray, posterior_samples: ndarray, n_closest: int = 1000
    ):
        xs = self.format_arr(x)
        posterior_samples = self.format_arr(posterior_samples)

        m = len(xs)
        ndraws = len(posterior_samples)
        xs_approx_pred_draws = np.zeros((ndraws, m))
        for i, sample in enumerate(posterior_samples):
            print(f"\rComputed {i+1} out of {ndraws} posterior draws", end="")
            approx_pred_means, approx_pred_vars = self.approx(
                x, np.exp(sample), n_closest
            )
            xs_approx_pred = approx_pred_means + np.sqrt(
                approx_pred_vars
            ) * np.random.randn(m)
            xs_approx_pred_draws[i] = xs_approx_pred
        print()
        return xs_approx_pred_draws


def test_interpolation():
    np.random.seed(1252149)
    n = (64, 64)
    mask = np.ones(n)

    n_missing = 10
    missing_idxs = np.random.randint(n[0], size=(n_missing, 2))
    mask[tuple(missing_idxs.T)] = 0.0
    m = mask.astype(bool)

    plt.imshow(mask, cmap="Greys", origin="lower")
    plt.show()

    grid = RectangularGrid(n)
    model = SquaredExponentialModel()
    model.rho = 10
    model.sigma = 1
    model.nugget = 1e-5
    sampler = SamplerOnRectangularGrid(model, grid)
    z_ = sampler()
    z = z_ * mask

    params = np.log([10.0, 1.0])

    grid = RectangularGrid(n, mask=m)
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=1e-5)
    dw.fit(None, prior=False)

    interp = SimpleKriging(z, RectangularGrid(n, mask=m), model)
    pred_means, pred_vars = interp(interp.missing_xs, params=np.exp(dw.res.x))
    approx_pred_means, approx_pred_vars = interp.approx(
        interp.missing_xs, params=np.exp(dw.res.x), n_closest=100
    )

    print(z_[~m].round(3), pred_means.round(3), approx_pred_means.round(3), sep="\n")

    # stop

    z[~m] = pred_means

    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
    ax[0].set_title("original", fontsize=22)
    im1 = ax[0].imshow(z_, cmap="Spectral", origin="lower")
    fig.colorbar(im1, shrink=0.5, ax=ax[0])

    ax[1].set_title("interpolated", fontsize=22)
    im2 = ax[1].imshow(z, cmap="Spectral", origin="lower")
    fig.colorbar(im2, shrink=0.5, ax=ax[1])
    fig.tight_layout()
    plt.show()
    # stop

    dewhittle_post, A = dw.RW_MH(200)  # unadjusted
    preds = interp.bayesian_prediction(interp.missing_xs, dewhittle_post)
    approx_preds = interp.approx_bayesian_prediction(
        interp.missing_xs, dewhittle_post, n_closest=100
    )

    density_labels = ["predictions", "approx predictions"]
    plot_marginals(
        [preds, approx_preds],
        shape=(2, 5),
        truths=z_[~m],
        density_labels=density_labels,
        title="posterior predictive densities",
    )


# interpolation_test()
