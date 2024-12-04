# Welcome to the documentation of Debiased Spatial Whittle! 

This documentation includes an API reference as well as some example
notebooks.

## Package description
This package implements a Fourier-based approximate likelihood method
to infer parametric spatio-temporal covariance models from large gridded data.
While the data need to sit on a grid, missing observations are permitted.

The package can be run with Cupy as a backend to benefit from GPU
implementations of the Fast Fourier Transform for additional perfomance
gains (order x100 versus CPU).