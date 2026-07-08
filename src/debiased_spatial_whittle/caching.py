from functools import lru_cache
import logging


def ban_if_frozen(func):

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "frozen") and self.frozen:
            raise ValueError("Not available as the instance is frozen")
        else:
            return func(self, *args, **kwargs)

    return wrapper


def lru_cache_frozen(func):
    cached = lru_cache(maxsize=3)(func)

    def wrapper(*args, **kwargs):
        frozen = True
        for arg in args:
            if hasattr(arg, "frozen") and not arg.frozen:
                frozen = False
        if frozen:
            logging.info(f"Using cached value - {func.__name__}")
            for arg in args:
                if hasattr(arg, "frozen"):
                    arg.callers.append(cached)
            return cached(*args, **kwargs)
        else:
            logging.info(f"Computing value. - {func.__name__}")
            return func(*args, **kwargs)

    return wrapper


class Freezable:
    def __init__(self):
        self.frozen = False
        self.parents = []
        self.callers = []
        super().__init__()

    def freeze(self):
        self.frozen = True
        for k, v in self.__dict__.items():
            if isinstance(v, Freezable):
                v.freeze()

    def unfreeze(self, recursive: bool = False):
        self.frozen = False
        self._delete_caches()
        for parent in self.parents:
            if isinstance(parent, Freezable):
                parent.unfreeze()
        if recursive:
            for k, v in self.__dict__.items():
                if isinstance(v, Freezable):
                    v.unfreeze(recursive=True)

    def _delete_caches(self):
        for caller in self.callers:
            caller.cache_clear()

    def __setattr__(self, key, value):
        if isinstance(value, Freezable):
            value.parents.append(self)
        super().__setattr__(key, value)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from debiased_spatial_whittle.models import SquaredExponentialModel
    from debiased_spatial_whittle.grids import RectangularGrid
    from debiased_spatial_whittle.inference import Periodogram, ExpectedPeriodogram

    g = RectangularGrid((32, 32))
    model = SquaredExponentialModel(rho=32)
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    ep(model)