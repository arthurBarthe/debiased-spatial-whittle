from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()

from progressbar import progressbar

from debiased_spatial_whittle.models.base import CovarianceModel
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.periodogram import ExpectedPeriodogram


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
        x, y = xp.meshgrid(xp.arange(n1), xp.arange(n2), indexing="ij")
        a = xp.logical_and(x < n1 - m1, y < n2 - m2)
        b = xp.logical_and(x >= -m1, y >= -m2)
        return xp.logical_and(a, b)

    @staticmethod
    def _get_indices_ep2(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = xp.meshgrid(xp.arange(n1), xp.arange(n2), indexing="ij")
        a = xp.logical_and(x >= m1, y >= m2)
        b = xp.logical_and(x < n1 + m1, y < n2 + m2)
        return xp.logical_and(a, b)

    @staticmethod
    def _get_indices_ep3(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = xp.meshgrid(xp.arange(n1), xp.arange(n2), indexing="ij")
        a = xp.logical_and(x <= m1, y <= m2)
        b = xp.logical_and(x >= m1 - n1 + 1, y >= m2 - n2 + 1)
        return xp.logical_and(a, b)

    @staticmethod
    def _get_indices_ep4(n, m):
        n1, n2 = n
        m1, m2 = m
        x, y = xp.meshgrid(
            xp.arange(n1 - 1, -1, -1), xp.arange(n2 - 1, -1, -1), indexing="ij"
        )
        a = xp.logical_and(x <= m1, y <= m2)
        b = xp.logical_and(x >= m1 - n1 + 1, y >= m2 - n2 + 1)
        return xp.logical_and(a, b)

    def exact_summation1(
        self,
        model: CovarianceModel,
        expected_periodogram: ExpectedPeriodogram,
        f: xp.ndarray = None,
        f2: xp.ndarray = None,
        return_terms: bool = False,
        normalize: bool = False,
    ):
        g = self.grid.mask
        n = g.shape
        n1, n2 = n
        ep = expected_periodogram(model)
        s = []
        if f is None:
            f = xp.ones_like(ep)
            f2 = xp.ones_like(ep)
        if normalize is True:
            ep = expected_periodogram(model)
        else:
            ep = xp.ones(n)
        for m1 in progressbar(range(-n1 + 1, n1)):
            for m2 in range(-n2 + 1, n2):
                ep_i1 = self._get_indices_ep1(n, (m1, m2))
                ep_i2 = self._get_indices_ep2(n, (m1, m2))
                seq_ep = ep[ep_i1] * ep[ep_i2]
                f_seq = f[ep_i1] * f2[ep_i2]
                seq = expected_periodogram.cov_diagonals(model, (m1, m2)).flatten()
                s.append(xp.sum(seq * f_seq / seq_ep))
        if return_terms:
            return xp.sum(s), xp.array(s)
        return xp.sum(s)

    def exact_summation2(
        self,
        model: CovarianceModel,
        expected_periodogram: ExpectedPeriodogram,
        f: xp.ndarray = None,
        f2: xp.ndarray = None,
        normalize: bool = True,
    ) -> float:
        n = self.grid.n
        n1, n2 = n
        ep = expected_periodogram(model)
        s = 0
        if f is None:
            f = xp.ones_like(ep)
            f2 = xp.ones_like(ep)
        if not normalize:
            ep = xp.ones_like(ep)
        for m1 in progressbar(range(2 * (n1 - 1) + 1)):
            for m2 in range(2 * (n2 - 1) + 1):
                ep_i1 = self._get_indices_ep3(n, (m1, m2))
                temp = ep[ep_i1]
                seq_ep = temp * xp.flip(
                    temp,
                    [
                        0,
                    ],
                )
                temp = f2[ep_i1]
                f_seq = f[ep_i1] * xp.flip(
                    temp,
                    [
                        0,
                    ],
                )
                seq = expected_periodogram.cov_antidiagonals(model, (m1, m2)).flatten()
                s += xp.sum(seq * f_seq)
        return s

    def fill_mat_separable1(
        self, model: CovarianceModel, expected_periodogram: ExpectedPeriodogram
    ):
        n = self.grid.n
        n1, n2 = n
        mat = xp.zeros((n1 * n2, n1 * n2))
        ep = expected_periodogram.cov_dft_diagonals(model, (0, 0))
        for m1 in range(-n1 + 1, n1):
            for m2 in range(-n2 + 1, n2):
                ep_i1 = self._get_indices_ep1(n, (m1, m2))
                ep_i2 = self._get_indices_ep2(n, (m1, m2))
                ij = xp.flatnonzero(ep_i1)
                ij2 = xp.flatnonzero(ep_i2)
                seq_ep = ep[ep_i1] * ep[ep_i2]
                seq = expected_periodogram.cov_diagonals(model, (m1, m2))
                q = m1 * n2 + m2
                d = xp.diag(mat, q)
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
