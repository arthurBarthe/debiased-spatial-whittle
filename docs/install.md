The package can be installed via one of the following methods.

### CPU-only

The package can be installed via one of the following methods.

1. Via the use of [Poetry](https://python-poetry.org/), by running the following command:

   ```bash
   poetry add debiased-spatial-whittle
   ```

2. Otherwise, you can directly install via pip:

    ```bash
    pip install debiased-spatial-whittle
    ```

### GPU
The Debiased Spatial Whittle likelihood relies on the Fast Fourier Transform (FFT) for computational efficiency.
GPU implementations of the FFT provide additional computational efficiency (order x100) at almost no additional cost thanks to GPU implementations of the FFT algorithm.

If you want to install with GPU dependencies (Cupy and Pytorch):

1. You need an NVIDIA GPU
2. You need to install the CUDA Toolkit. See for instance Cupy's [installation page](https://docs.cupy.dev/en/stable/install.html).
3. You can install Cupy or pytorch yourself in your environment. Or you can specify an extra to poetry, e.g.

   ```bash
   poetry add debiased-spatial-whittle -E gpu12
   ```
   if you version of the CUDA toolkit is 12.* (use gpu11 if your version is 11.*)

One way to check your CUDA version is to run the following command in a terminal:

```bash
   nvidia-smi
```

You can then switch to using e.g. Cupy instead of numpy as the backend via:

```python
from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend("cupy")
  ```

This should be run before any other import from the debiased_spatial_whittle package.
