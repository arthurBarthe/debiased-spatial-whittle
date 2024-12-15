# Spatial Debiased Whittle Likelihood

![Image](logo.png)

[![Documentation Status](https://readthedocs.org/projects/debiased-spatial-whittle/badge/?version=latest)](https://debiased-spatial-whittle.readthedocs.io/en/latest/?badge=latest)
[![.github/workflows/run_tests_on_push.yaml](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/run_tests_on_push.yaml/badge.svg)](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/run_tests_on_push.yaml)
[![Pypi](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/pypi.yml/badge.svg)](https://github.com/arthurBarthe/debiased-spatial-whittle/actions/workflows/pypi.yml)

## Introduction

This package implements the Spatial Debiased Whittle Likelihood (SDW) as presented in the article of the same name, by the following authors:

- Arthur P. Guillaumin
- Adam M. Sykulski
- Sofia C. Olhede
- Frederik J. Simons

The SDW extends ideas from the Whittle likelihood and Debiased Whittle Likelihood to random fields and spatio-temporal data. In particular, it directly addresses the bias issue of the Whittle likelihood for observation domains with dimension greater than 2. It also allows us to work with rectangular domains (i.e., rather than square), missing observations, and complex shapes of data.

## Installation instructions

The package can be installed via one of the following methods.

1. Via the use of Poetry ([https://python-poetry.org/](https://python-poetry.org/)), by running the following command:

   ```bash
   poetry add debiased-spatial-whittle
   ```

2. Otherwise, you can directly install via pip:

    ```bash
    pip install debiased-spatial-whittle
    ```

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
