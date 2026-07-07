from debiased_spatial_whittle.backend import BackendManager
xp = BackendManager.get_backend()

from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.univariate import ExponentialModel
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.likelihood import DebiasedWhittle


def test_jmat():
    g = RectangularGrid((16, 16))
    p = Periodogram()
    ep = ExpectedPeriodogram(g, p)
    d = DebiasedWhittle(p, ep)
    model = ExponentialModel(rho=10, sigma=1)
    jmat = d.jmatrix(model, model.parameter_names)
    print(jmat)
    assert xp.all(xp.diag(jmat) >= 0)