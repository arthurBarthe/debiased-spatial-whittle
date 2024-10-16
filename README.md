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

The package can be installed via one of the following methods. Note that in all cases, since the repository is currently private, git needs to be configured on your machine with an SSH key linking your machine to your GitHub account.

1. Via the use of Poetry ([https://python-poetry.org/](https://python-poetry.org/)), by adding the following line to the dependencies listed in the `pyproject.toml` of your project:

    ```toml
    debiased-spatial-whittle = {git = "git@github.com:arthurBarthe/dbw_private.git", branch="master"}
    ```

2. Otherwise, you can directly install via pip:

    ```bash
    pip install git+https://github.com/arthurBarthe/dbw_private.git
    ```

3. Install for development - in this case, you need to clone this repo and run

    ```bash
    poetry install
    ```

    in a terminal from where you cloned the repository.

If you get an error message regarding the version of Python, install a compatible version of Python on your machine and point to it via

```bash
poetry env use <path_to_python>
