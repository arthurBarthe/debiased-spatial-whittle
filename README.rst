===================================
Spatial Debiased Whittle Likelihood
===================================

.. image:: logo.png
    :width: 150
    :alt: Image

.. image:: https://readthedocs.org/projects/debiased-spatial-whittle/badge/?version=latest
    :target: https://debiased-spatial-whittle.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Introduction
============
This package implements the Spatial Debiased Whittle Likelihood (SDW) as presented
in the article of the same name, by the following authors,

* Arthur P. Guillaumin,
* Adam M. Sykulski,
* Sofia C. Olhede,
* Frederik J. Simons.

The SDW extends ideas from the Whittle likelihood and Debiased
Whittle Likelihood to random fields and spatio-temporal data.
In particular it directly addresses the bias issue of the Whittle
likelihood for observation domains with dimension greater than 2.
It also allows us to work with rectangular domains (i.e. rather than square),
missing observations, and complex shapes of data.


Installation instructions
=========================
The package can be installed via one of the following methods. Note that in
all cases, since the repository is currently private, git needs to be configured
on your machine with an SSH key linking your machine to your github account.

1. Via the use of Poetry (https://python-poetry.org/), by adding
the following line to the dependencies listed in the pyproject.toml
of your project:


..  code-block:: toml

    debiased-spatial-whittle = {git = "git@github.com:arthurBarthe/dbw_private.git", branch="master"}

2. Otherwise you can directly install via pip:

.. code-block:: bash

    pip install git+https://github.com/arthurBarthe/dbw_private.git

3. Install for developement - in this case you need to clone this repo and
run

.. code-block:: bash

    poetry install

in a terminal from where you cloned the repository.

If you get an error message regarding the version of python, install
a compatible version of python on your machine and point to it via

.. code-block:: bash

    poetry env use <path_to_python>

before running the poetry install command.



Documentation
The documentation_ is hosted on readthedocs. It contains an API reference as well as
examples.

.. _documentation: https://debiased-spatial-whittle.readthedocs.io/en/latest/


Tips
====
Tapering
-----------
While tapering is not necessary for consistency of the SDW, it can be
usefull for finite sample size in order to reduce variance, when
remaining boundary effects are still important. In particular this
is true for spectral models with a strong dynamic range, such as
the squared exponential covariance model.
