# Spatial Debiased Whittle Likelihood

![Image](logo.png)

[![Documentation Status](https://readthedocs.org/projects/debiased-spatial-whittle/badge/?version=latest)](https://debiased-spatial-whittle.readthedocs.io/en/latest/?badge=latest)
[![.github/workflows/run_tests_on_push.yaml](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/run_tests_on_push.yaml/badge.svg)](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/run_tests_on_push.yaml)
[![Pypi](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/pypi.yml/badge.svg)](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/pypi.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arthurBarthe/debiased-spatial-whittle/master)

## Introduction

This package implements the Spatial Debiased Whittle Likelihood (SDW) as presented in the article of the same name, by the following authors:

- Arthur P. Guillaumin
- Adam M. Sykulski
- Sofia C. Olhede
- Frederik J. Simons

The SDW extends ideas from the Whittle likelihood and Debiased Whittle Likelihood to random fields and spatio-temporal data. In particular, it directly addresses the bias issue of the Whittle likelihood for observation domains with dimension greater than 2. It also allows us to work with rectangular domains (i.e., rather than square), missing observations, and complex shapes of data.

## Installation instructions

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

You can then switch to using e.g. Cupy instead of numpy as the backend via:

   ```python
    from debiased_spatial_whittle.backend import BackendManager
    BackendManager.set_backend("cupy")
   ```

This should be run before any other import from the debiased_spatial_whittle package.

## Development

Firstly, you need to install poetry. Then, git clone this repository, ad run the following command from
the directory corresponding to the package.

   ```bash
   poetry install
   ```

If you run into some issue regarding the Python version, you can run
   ```bash
   poetry env use <path_to_python>
   ```
where <path_to_python> is the path to a Python version compatible with the requirements in pyproject.toml.

### Unit tests
Unit tests are run with pytest. On Pull-requests, the unit tests will be
run.

## Documentation
The documentation is hosted on readthedocs. It is based on docstrings.
Currently, it points to the joss_paper branch and is updated on any push to that branch.

## Versioning
Currently, versioning is handled manuallyusing poetry, e.g.

   ```bash
   poetry version patch
   ```
or
   ```bash
   poetry version minor
   ```

When creating a release in Github, the version tag should be set to match
the version in th pyproject.toml. Creating a release in Github will trigger
a Github workflow that will publish to Pypi (see Pypi section).

## PyPi
The package is updated on PyPi automatically on creation of a new
release in Github. Note that currently the version in pyproject.toml
needs to be manually updated. This should be fixed by adding
a step in the workflow used for publication to Pypi.
