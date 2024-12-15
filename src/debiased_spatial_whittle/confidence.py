from typing import Tuple

from debiased_spatial_whittle.backend import BackendManager

np = BackendManager.get_backend()

from numpy.random import randint, rand
from progressbar import progressbar
import matplotlib.pyplot as plt

from .models import CovarianceModel
from .grids import RectangularGrid
from .periodogram import ExpectedPeriodogram


class CovarianceFFT:
    def __init__(self, grid: RectangularGrid):
        self.grid = grid

    @property
    def g(self):
        return self.grid.mask

    @staticmethod
    def _get_indices_ep1(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        a = np.logical_and(x < n1 - m1, y < n2 - m2)
        b = np.logical_and(x >= -m1, y >= -m2)
        return np.logical_and(a, b)

    @staticmethod
    def _get_indices_ep2(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        a = np.logical_and(x >= m1, y >= m2)
        b = np.logical_and(x < n1 + m1, y < n2 + m2)
        return np.logical_and(a, b)

    @staticmethod
    def _get_indices_ep3(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        a = np.logical_and(x <= m1, y <= m2)
        b = np.logical_and(x >= m1 - n1 + 1, y >= m2 - n2 + 1)
        return np.logical_and(a, b)

    @staticmethod
    def _get_indices_ep4(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = np.meshgrid(
            np.arange(n1 - 1, -1, -1), np.arange(n2 - 1, -1, -1), indexing="ij"
        )
        a = np.logical_and(x <= m1, y <= m2)
        b = np.logical_and(x >= m1 - n1 + 1, y >= m2 - n2 + 1)
        return np.logical_and(a, b)

    def exact_summation1(
        self,
        model: CovarianceModel,
        expected_periodogram: ExpectedPeriodogram,
        f: np.ndarray = None,
        f2: np.ndarray = None,
        return_terms: bool = False,
        normalize: bool = False,
    ):
        g = self.grid.mask
        n = g.shape
        n1, n2 = n
        ep = expected_periodogram(model)
        s = []
        if f is None:
            f = np.ones_like(ep)
            f2 = np.ones_like(ep)
        if normalize is True:
            ep = expected_periodogram(model)
        else:
            ep = np.ones(n)
        for m1 in progressbar(range(-n1 + 1, n1)):
            for m2 in range(-n2 + 1, n2):
                ep_i1 = self._get_indices_ep1(n, (m1, m2))
                ep_i2 = self._get_indices_ep2(n, (m1, m2))
                seq_ep = ep[ep_i1] * ep[ep_i2]
                f_seq = f[ep_i1] * f2[ep_i2]
                seq = expected_periodogram.cov_diagonals(model, (m1, m2)).flatten()
                s.append(np.sum(seq * f_seq / seq_ep))
        if return_terms:
            return np.sum(s), np.array(s)
        return np.sum(s)

    def exact_summation2(
        self,
        model: CovarianceModel,
        expected_periodogram: ExpectedPeriodogram,
        f: np.ndarray = None,
        f2: np.ndarray = None,
        normalize: bool = True,
    ) -> float:
        n = self.grid.n
        n1, n2 = n
        ep = expected_periodogram(model)
        s = 0
        if f is None:
            f = np.ones_like(ep)
            f2 = np.ones_like(ep)
        if not normalize:
            ep = np.ones_like(ep)
        for m1 in progressbar(range(2 * (n1 - 1) + 1)):
            for m2 in range(2 * (n2 - 1) + 1):
                ep_i1 = self._get_indices_ep3(n, (m1, m2))
                temp = ep[ep_i1]
                seq_ep = temp * np.flip(
                    temp,
                    [
                        0,
                    ],
                )
                temp = f2[ep_i1]
                f_seq = f[ep_i1] * np.flip(
                    temp,
                    [
                        0,
                    ],
                )
                seq = expected_periodogram.cov_antidiagonals(model, (m1, m2)).flatten()
                s += np.sum(seq * f_seq)
        return s

    def fill_mat_separable1(
        self, model: CovarianceModel, expected_periodogram: ExpectedPeriodogram
    ):
        n = self.grid.n
        n1, n2 = n
        mat = np.zeros((n1 * n2, n1 * n2))
        ep = expected_periodogram.cov_dft_diagonals(model, (0, 0))
        for m1 in range(-n1 + 1, n1):
            for m2 in range(-n2 + 1, n2):
                ep_i1 = self._get_indices_ep1(n, (m1, m2))
                ep_i2 = self._get_indices_ep2(n, (m1, m2))
                ij = np.flatnonzero(ep_i1)
                ij2 = np.flatnonzero(ep_i2)
                seq_ep = ep[ep_i1] * ep[ep_i2]
                seq = expected_periodogram.cov_diagonals(model, (m1, m2))
                q = m1 * n2 + m2
                d = np.diag(mat, q)
                d.setflags(write=True)
                try:
                    if q <= 0:
                        d[ij2] = seq.flatten() / seq_ep.flatten()
                    else:
                        d[ij] = seq.flatten() / seq_ep.flatten()
                except Exception as e:
                    print(m1, m2, q, ij, ij2)
                    raise e
        return mat


##################################################################################
##################################################################################
############################### MCMC #############################################
##################################################################################
##################################################################################


class AppDiags:
    def __init__(
        self,
        model: CovarianceModel,
        expected_periodogram: ExpectedPeriodogram,
        f=None,
        f2=None,
    ):
        self.g = expected_periodogram.grid.mask
        self.n = self.g.shape
        self.f = f
        self.f2 = f2
        self.model = model
        self.expected_periodogram = expected_periodogram
        if f is None:
            self.f = np.ones_like(self.g)
            self.f2 = np.ones_like(self.g)
        self.ep = expected_periodogram.cov_dft_diagonals(model, (0, 0))

    # TODO duplicate code
    @staticmethod
    def _get_indices_ep1(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        a = np.logical_and(x < n1 - m1, y < n2 - m2)
        b = np.logical_and(x >= -m1, y >= -m2)
        return np.logical_and(a, b)

    @staticmethod
    def _get_indices_ep2(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        a = np.logical_and(x >= m1, y >= m2)
        b = np.logical_and(x < n1 + m1, y < n2 + m2)
        return np.logical_and(a, b)

    def __call__(self, state: Tuple[int, int]):
        m1, m2 = state
        seq = self.expected_periodogram.cov_diagonals(self.model, (m1, m2)).flatten()
        ep_i1 = self._get_indices_ep1(self.n, (m1, m2))
        ep_i2 = self._get_indices_ep2(self.n, (m1, m2))
        seq_ep = self.ep[ep_i1] * self.ep[ep_i2]
        f = self.f[ep_i1] * self.f2[ep_i2]
        temp = seq / seq_ep
        temp2 = np.sum(temp)
        # todo add other term
        temp3 = np.sum(temp * f) / temp2
        return temp2, temp3


class McmcDiags:
    def __init__(
        self,
        model: CovarianceModel,
        expected_periodogram: ExpectedPeriodogram,
        f: np.ndarray = None,
        f2: np.ndarray = None,
    ):
        self.app = AppDiags(model, expected_periodogram, f, f2)
        # TODO ugly (2 lines)
        self.g = expected_periodogram.grid.mask
        self.f, self.f2 = f, f2
        self.n = self.g.shape
        self.model = model
        self.current = randint(self.n[0]), randint(self.n[1])
        self.current = (0, 0)
        self.current_p, self.current_f = self.evaluate_p(self.current)
        self.proposal_width = max(2, self.n[0] // 20)
        self.history = []
        self.p_history = []
        self.f_history = []
        self.accept_history = []

    @property
    def acceptance_rate(self):
        return np.mean(self.accept_history)

    def propose(self):
        """Proposes a new state, given the current state"""
        m1, m2 = self.current
        new_m1 = m1 + randint(-self.proposal_width, self.proposal_width + 1)
        new_m1 = new_m1 % self.n[0]
        new_m2 = m2 + randint(-self.proposal_width, self.proposal_width + 1)
        new_m2 = new_m2 % self.n[1]
        return (new_m1, new_m2)

    def step(self):
        """Carries out a single step of the MCMC"""
        new = self.propose()
        new_p, new_f = self.evaluate_p(new)
        ratio = new_p / self.current_p
        accept = rand() <= ratio
        self.accept_history.append(accept)
        if accept:
            self.current = new
            self.current_p = new_p
            self.current_f = new_f
        self.history.append(self.current)
        self.p_history.append(
            (self.n[0] - self.current[0])
            * (self.n[1] - self.current[1])
            / self.current_p
        )
        self.f_history.append(self.current_f)
        return self.current_p * self.current_f

    def run(self, n_steps: int):
        """Runs the MCMC for n_steps steps"""
        for i_step in progressbar(range(n_steps)):
            self.step()

    def evaluate_p(self, state):
        """Evaluates the probability of a state, which is the correlation between
        the periodogram at two frequencies. Note that we give probability zero to
        the diagonal and anti-diagonal as these are known to be one and are processed separately
        from the MCMC."""
        if state[0] == 0 and state[1] == 0:
            return 0.0, None
        return self.app(state)

    def plot_trace(self, n_steps: int = None):
        """Plots a trace of all visited states"""
        if n_steps is None:
            n_steps = len(self.p_history)
        fig = plt.figure()
        ax = fig.add_subplot()
        history = np.array(self.history)
        history_x = history[:n_steps, 0]
        history_y = history[:n_steps, 1]
        ax.scatter(history_x, history_y, c="orange")
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)

    def partition_function(self):
        # TODO incorrect, we should not add 2 * self.n at the end.
        """Computes the estimate of the partition function"""
        return (
            self.n[0]
            * self.n[1]
            * (self.n[0] - 1)
            * (self.n[1] - 1)
            / np.mean(np.array(self.p_history))
        )

    def partition_function_trace(self):
        n_steps = len(self.history)
        # result = (2 * self.n[0] - 1) * (2 * self.n[1] - 1) * np.arange(1, n_steps +1) / np.cumsum(np.array(self.p_history))
        result = np.arange(1, n_steps + 1) / np.cumsum(np.array(self.p_history), 0)
        result *= self.n[0] * self.n[1] * (self.n[0] - 1) * (self.n[1] - 1)
        return result

    def estimate(self):
        """Compute the estimate of interest"""
        off_diagonal = np.mean(np.array(self.f_history)) * self.partition_function()
        on_diagonal = np.sum(self.f * self.f2)
        print(on_diagonal, off_diagonal)
        return on_diagonal + off_diagonal
