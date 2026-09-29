# Example: sampling from a ConvolModel defined by an empirical autocorrelation

# ##Imports
import numpy as np
import matplotlib.pyplot as plt

from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("numpy")
xp = BackendManager.get_backend()

from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.models.convol import ConvolModel
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid

# ##Grid and sample specification
# Unit deltas, shape (512, 512)
n = (1024, 1024)
grid = RectangularGrid(n)

# Build the sample: exp(-sqrt(i1^2 + i2^2) / 20)
i1, i2 = xp.meshgrid(
    xp.arange(n[0], dtype=xp.float64), xp.arange(n[1], dtype=xp.float64),
    indexing="ij",
)
sample = (np.cos(i1 / 5) + np.sin(i2 / 15)) * np.exp(-np.abs(i1 / 10) - np.abs(i2 / 20))
sample = sample / np.max(np.abs(sample))

# ##Model creation
model = ConvolModel(grid, sample=sample)

# ##Sampling
sampler = SamplerOnRectangularGrid(model, grid)
simulated = sampler()
print(f"Simulated sample shape: {simulated.values.shape}")

# ##Plot
plt.imshow(simulated.values, origin="lower", cmap="Spectral")
plt.colorbar(label="value")
plt.title("Sample from ConvolModel")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
