===================================
Spatial Debiased Whittle Likelihood
===================================

.. image:: logo.png
    :width: 150
    :alt: Image


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



Documentation
=============
The documentation currently only relies on Doc strings, and some examples that show the
breadth of real-World scenarios for which the method can be applied.

1. A standard full rectangular grid
2. A circular observation domain
3. Bernoulli missing observations
4. Data on a France-like shape
5. A mix of the last two
6. A non-isotropic exponential covariance model
7. A full Matern covariance model


Tips
====
Tapering
-----------
While tapering is not necessary for consistency of the SDW, it can be
usefull for finite sample size in order to reduce variance, when
remaining boundary effects are still important. In particular this
is true for spectral models with a strong spectral slope, such as
the squared exponential covariance model.


Future versions
===============
For future versions we will implement:

1. Interpolation using the estimated covariance model
