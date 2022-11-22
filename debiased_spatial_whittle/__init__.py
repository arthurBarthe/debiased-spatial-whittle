__version__ = '0.1.2'

from .likelihood import fit, periodogram
from .cov_funcs import exp_cov, sq_exp_cov, matern15_cov_func, exp_cov_anisotropic, matern, exp_cov_separable
from .expected_periodogram import compute_ep
from .simulation import sim_circ_embedding