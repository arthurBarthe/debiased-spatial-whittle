from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

from scipy.special import gamma, kv


def exp_cov(lags, rho, sigma=1.0):
    return sigma**2 * np.exp(-np.sqrt(sum((lag**2 for lag in lags))) / rho)


def exp_cov_anisotropic(lags, rho_1, rho_2, theta=0.0, sigma=1.0):
    lag_0 = lags[0] * np.cos(theta) - lags[1] * np.sin(theta)
    lag_1 = lags[0] * np.sin(theta) + lags[1] * np.cos(theta)
    norm = np.sqrt((lag_0 / rho_1) ** 2 + (lag_1 / rho_2) ** 2)
    return sigma**2 * np.exp(-norm)


def exp_cov1d(x: np.ndarray, rho: float):
    return np.exp(-np.abs(x) / rho) + 0.00 * (x == np.zeros_like(x))


def sq_exp_cov1d(x: np.ndarray, rho: float):
    return np.exp(-(x**2) / rho**2) + 0.00 * (x == np.zeros_like(x))


def exp_cov_separable(lags, rho, sigma=1.0):
    return exp_cov1d(lags[0], rho) * exp_cov1d(lags[1], rho)


def sq_exp_cov_separable(lags, rho, sigma=1.0):
    return sq_exp_cov1d(lags[0], rho) * sq_exp_cov1d(lags[1], rho)


def matern15_cov_func(lags, rho, sigma=1.0):
    K = np.sqrt(3) * np.sqrt(sum((lag**2 for lag in lags))) / rho
    return (1.0 + K) * np.exp(-K)


def sq_exp_cov(lags, rho, sigma=1.0):
    lags = np.stack(lags)
    return sigma**2 * np.exp(
        -(sum((lag**2 for lag in lags))) / (2 * rho**2)
    ) + 0.001 * (np.all(lags == 0, axis=0))


def matern(lags, rho, nu, sigma=1.0):
    d = np.sqrt(sum((lag**2 for lag in lags)))
    term1 = 2 ** (1 - nu) / gamma(nu)
    term2 = (np.sqrt(2 * nu) * d / rho) ** nu
    term3 = kv(nu, np.sqrt(2 * nu) * d / rho)
    val = sigma**2 * term1 * term2 * term3
    val[d == 0] = sigma**2
    return val


## Derivatives with respect to parameters


def exp_cov_prime(lags, rho, sigma=1.0):
    d_rho = exp_cov(lags, rho, sigma) * np.sqrt(sum((lag**2 for lag in lags))) / rho**2
    return (d_rho,)


def sq_exp_cov_prime(lags, rho, sigma=1.0):
    d_rho = sq_exp_cov(lags, rho, sigma) * (sum((lag**2 for lag in lags))) / rho**3
    return (d_rho,)


def matern15_cov_func_prime(lags, rho, sigma=1.0):
    d_rho = (
        matern15_cov_func(lags, rho, sigma)
        * np.sqrt(3)
        * np.sqrt(sum((lag**2 for lag in lags)))
        / rho**2
    )
    return (d_rho,)
