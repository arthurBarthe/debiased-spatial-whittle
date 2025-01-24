from debiased_spatial_whittle.backend import BackendManager

BackendManager.set_backend("torch")
np = BackendManager.get_backend()
import numpy

import matplotlib.pyplot as plt
import scipy.linalg
import torch

import debiased_spatial_whittle.grids as grids
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.likelihood import Estimator, DebiasedWhittle

model = ExponentialModel()
model.rho = 20
model.sigma = 1
model.nugget = 0.0

m = 64
shape = (m * 1, m * 1)

x_0, y_0, diameter = m // 2, m // 2, m
x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
circle = ((x - x_0) ** 2 + (y - y_0) ** 2) <= 1 / 4 * diameter**2
circle = circle * 1.0
grid_circle = RectangularGrid(shape)
# grid_circle.mask = circle
sampler = SamplerOnRectangularGrid(model, grid_circle)

z = sampler()

periodogram = Periodogram()
expected_periodogram = ExpectedPeriodogram(grid_circle, periodogram)
debiased_whittle = DebiasedWhittle(periodogram, expected_periodogram)

from torch.nn import Module, Sequential, ReLU, Softplus, Linear, BatchNorm1d

s1 = 256
nn = Sequential(
    Linear(3, 2 * s1),
    ReLU(),  # BatchNorm1d(2 * s1),
    Linear(2 * s1, 2 * s1),
    ReLU(),  # BatchNorm1d(2 * s1),
    Linear(2 * s1, 2 * s1),
    ReLU(),  # BatchNorm1d(2 * s1),
    Linear(2 * s1, s1),
    ReLU(),  # BatchNorm1d(s1),
    Linear(s1, s1),
    ReLU(),  # BatchNorm1d(s1),
    Linear(s1, 1),
)
nn.train()

from debiased_spatial_whittle.models import AliasedSpectralModel, Parameters
from debiased_spatial_whittle.models import Parameter


class NNmodel(AliasedSpectralModel):
    def __init__(self, nn: Module):
        rho = Parameter("rho", (0, 20))
        params = Parameters([rho])
        super(NNmodel, self).__init__(params)
        self.nn = nn

    def spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        out_shape = frequencies.shape[:-1]
        ndim = frequencies.ndim - 1
        # reshape
        frequencies = torch.flatten(frequencies, end_dim=-2)
        # add parameter value
        rho = (self.rho.value - 10) / 10
        frequencies1 = torch.cat(
            (frequencies, rho * torch.ones((frequencies.shape[0], 1))), dim=-1
        )
        frequencies2 = torch.cat(
            (-frequencies, rho * torch.ones((frequencies.shape[0], 1))), dim=-1
        )
        sdf = torch.exp((nn(frequencies1) + nn(frequencies2)) / 2)
        sdf = sdf / np.mean(sdf) / (2 * np.pi) ** ndim * 1
        sdf = np.reshape(sdf, out_shape)
        return sdf

    def train(self, z):
        pass

    def _gradient(self, x: np.ndarray):
        raise NotImplementedError()


model_est = NNmodel(nn)


from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

adam = Adam(nn.parameters(), lr=1e-3)

scheduler = ReduceLROnPlateau(adam, patience=7, factor=0.1, min_lr=1e-5)

for i in range(10000):
    nn.train()
    adam.zero_grad()
    loss = 0
    # generate a sample
    import random

    for j in range(10):
        rho = random.random() * 15 + 5
        model.rho = rho
        model_est.rho = rho
        sampler = SamplerOnRectangularGrid(model, grid_circle)
        z = sampler()
        #
        mask = np.zeros_like(z)
        mask[
            torch.randint(0, z.shape[0], (1000,)), torch.randint(0, z.shape[0], (1000,))
        ] = 1
        # debiased_whittle.frequency_mask = mask
        loss_j = debiased_whittle(z, model_est)
        loss_j = loss_j
        loss_j.backward()
        loss += loss_j
    print(i, loss)
    # loss.backward()
    adam.step()
    scheduler.step(loss)
    print("Learning rate: ", scheduler.get_last_lr())
    # plot
    rho = random.random() * 0 + 20
    model_est.rho = rho
    #
    frequencies = grid_circle.fourier_frequencies
    nn.eval()
    with torch.no_grad():
        sdf = model_est.spectral_density(frequencies).numpy()
    try:
        plt.close()
    except:
        pass
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(10 * numpy.fft.fftshift(numpy.log10(sdf)), cmap="Spectral_r")
    plt.colorbar()
    plt.title(f"{rho=}")
    plt.subplot(1, 4, 2)

    with torch.no_grad():
        ep_plot = expected_periodogram(model_est)
        res = 1 - np.exp(-periodogram(z) / ep_plot)
    plt.imshow(numpy.fft.fftshift(res), vmin=0, vmax=1)
    plt.colorbar()
    plt.subplot(1, 4, 3)
    plt.hist(res.flatten(), bins=np.linspace(0, 1, 10))
    plt.subplot(1, 4, 4)
    plt.imshow(10 * (numpy.fft.fftshift(numpy.log10(ep_plot))), cmap="Spectral_r")
    # plt.imshow(model_est.call_on_rectangular_grid(grid_circle).detach())
    # plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)

    #
    if i % 10 == 0:
        with torch.no_grad():
            print("----------")
            rho = 18
            print(rho)
            model.rho = rho
            grid2 = RectangularGrid((2 * m, 2 * m))
            sampler = SamplerOnRectangularGrid(model, grid2)
            ep2 = ExpectedPeriodogram(grid2, periodogram)
            dbw2 = DebiasedWhittle(periodogram, ep2)
            z = sampler()
            values = []
            for rho in range(5, 20):
                model_est.rho = rho
                values.append(dbw2(z, model_est).item())
            print(numpy.argsort(values) + 5)
