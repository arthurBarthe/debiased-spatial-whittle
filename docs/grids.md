## grids.py
This module provides tools to define sampling grids. Currently, the code
is restricted to orthogonal grids. This is not a limitation inherent to the proposed
method, and it would not be very hard to generalize to non-orthogonal grids.

::: debiased_spatial_whittle.grids.RectangularGrid


## simulation.py
This module provides tools to efficiently sample from covariance models on
grids using circulant embeddings and the Fast Fourier Transform.

::: debiased_spatial_whittle.simulation.SamplerOnRectangularGrid

::: debiased_spatial_whittle.simulation.SamplerBUCOnRectangularGrid


## spatial_kernel.py
This module provides the function that allows to compute the kernel of a grid of observations.

::: debiased_spatial_whittle.spatial_kernel.spatial_kernel
