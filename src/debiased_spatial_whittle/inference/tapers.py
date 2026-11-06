from typing import Callable

from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.caching import Freezable, ban_if_frozen, lru_cache_frozen

xp = BackendManager.get_backend()
ones = BackendManager.get_ones()
hanning = BackendManager.get_hanning()


class Taper(Freezable):
    def __init__(self, taper_func: Callable[[tuple[int] ], xp.ndarray]):
        self.taper_func = taper_func
        super().__init__()
        self.freeze()

    @property
    def taper_func(self):
        return self._taper_func

    @taper_func.setter
    @ban_if_frozen
    def taper_func(self, func):
        self._taper_func = func

    @lru_cache_frozen
    def __call__(self, shape: tuple[int]):
        return self.taper_func(shape)


class ConstantTaper(Taper):
    def __init__(self):
        func = lambda shape: ones(shape)
        super().__init__(func)


class HanningTaper(Taper):
    def __init__(self):

        def func(shape: tuple[int]):
            values = [hanning(n_i) for n_i in shape]
            grids = xp.meshgrid(*values, indexing="ij")
            out = grids[0]
            for i in range(1, len(shape)):
                out = out * grids[i]
            return out

        super().__init__(func)
