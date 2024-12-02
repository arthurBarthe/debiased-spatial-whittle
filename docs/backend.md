We provide a BackendManager class which allows switching between
Numpy, Cupy and PyTorch for array computations.
In particular, this means that all the FFTs carried out by the package
may benefit from GPU implementations provided by Cupy and Pytorch, 
leading to large performance gains.

Setting the backend must be carried out
before any other import from the package. Note that Cupy will only work
with a GPU.

```python
from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('numpy') # or 'cupy', or 'torch'
# if you want to use the GPU with pytorch
BackendManager.device = 'cuda:0'
# xp will either be numpy, cupy or torch depending on the set backend.
xp = BackendManager.get_backend()
```

If you need to move an array from GPU to CPU (for instance to carry
out a plot via matplotlib), you can use:

```python
a = xp.to_cpu(a)
```

If the backend is set to numpy, to_cpu will do nothing.