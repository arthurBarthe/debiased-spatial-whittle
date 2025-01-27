---
title: 'DSWL package: a Python implementation of the Debiased Spatial Whittle Likelihood'
tags:
  - Python
  - spatial
  - spatio-temporal
  - likelihood
  - covariance modelling
  - gaussian processes
authors:
  - name: Arthur P. Guillaumin
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Adam M. Sykulski
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Sofia C. Olhede
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Frederik J. Simons
    orcid: 0000-0000-0000-0000
    affiliation: 4
affiliations:
 - name: Queen Mary University of London, United Kingdom
   index: 1
 - name: Imperial College London, United Kingdom
   index: 2
 - name: Ecole Polytechnique Fédérale de Lausanne, Switzerland
   index: 3
 - name: Princeton University, United States of America
date: 20 January 2025
bibliography: paper.bib
---

# Summary
The Debiased Spatial Whittle Likelihood (DSWL) package is an open-source Python
package that implements the eponym paper [@guillaumin_debiased_2022].
The methodology allows users to efficiently infer the parameters of stationary
spatial and
spatio-temporal covariance models for univariate or multivariate processes from gridded data.
It leverages the Fast Fourier Transform, and therefore can benefit from further computational
gains from GPU implementations offered by PyTorch or Cupy, both made available within
the package as alternative backends to Numpy. As such, DSWL on GPU allows to fit
covariance models to data observed on grids with tens of millions of locations.

# Statement of need
Describing patterns of spatial and spatio-temporal covariance is of interest to practitioners in
a wide range of applied sciences such as geosciences, meteorology or climate
science. Stationary covariance modelling allows for a first-order approximation
of the covariance structure, and leads to many practical applications such as
krigging and forecasting via the conditional Gaussian multivariate
distribution.

A major hurdle in spatio-temporal modelling is the computational cost of the
Gaussian likelihood function. This is particularly relevant for modern spatio-temporal
datasets, from physics simulations to real-word data.
This computational burden also arises from complex
spatio-temporal covariance models with a large number of parameters
which typically require a high number of likelihood evaluations during the optimization process or
when running an MCMC sampler.

A common means to circumvent this computational burden is to use approximations to the Gaussian likelihood.
Among these,
the Whittle likelihood is a standard spectral domain method for gridded data.
Along its computational benefits, the Whittle likelihood provides robustness to departures
from Gaussianity and allows to restrict the second-order model to a specific range
of spatio-temporal frequencies.
However, for spatial and spatio-temporal data where $d\geq 2$, the standard Whittle likelihood
suffers from a large bias
and typically does not allow for missing observations.

`DSWL` is a Python implementation of the Debiased Spatial Whittle likelihood
[@guillaumin_debiased_2022], a method that addresses the bias of the Whittle likelihood
[@sykulski_debiased_2019].
While its use of the Fast Fourier Transform requires gridded data, the implemented
method additionally allows for missing observations, making it amenable to practical
applications where a full hypercube of data measurements might not
be available.
The package allows to treat the case of multivariate data, including where the
missingness patterns might differ between two variates.
The code base also includes tapering, the use of which can further
help alleviate boundary effects[@dahlhaus_edge_1987]. Finally, the user can switch between several backends,
Numpy, Cupy and PyTorch. This allows to further benefit from computational
gains via GPU implementations of the Fast Fourier Transform.

# Software structure

The software is organized around several modules that can be grouped into the following
categories:

- grids and sampling:
  - grids.py: allows to define the rectangular grids where the data sit via the
  class RectangularGrid. A mask of zeros (missing) and ones (not missing) can be
  set to specify potential missing observations.
  - simulation.py: allows to sample a realization from a model on a grid
- models:
  - models.py: allows to define a covariance model.
    Standard covariance models are pre-defined, such as the exponential
    covariance model, the squared exponential covariance model and
    the Matern covariance model.
- estimation:
  - periodogram.py: allows to compute the periodogram of the data, and to obtain
    the expected periodogram for a given model, grid, and periodogram combination.
  - multivariate_periodogram.py: allows to compute the periodogram for multivariate data.
  - likelihood.py: allows to define the Debiased Whittle Likelihood and the corresponding
    estimator.

A [documentation](https://debiased-spatial-whittle.readthedocs.io/en/latest/index.html)
including example notebooks is available, and issues can be raised on
[Github](https://github.com/arthurBarthe/debiased-spatial-whittle). Example notebooks can also be run directly in the browser
via [mybinder.org](https://mybinder.org/v2/gh/arthurBarthe/debiased-spatial-whittle/master).

# Acknowledgements
This research utilised Queen Mary's Apocrita HPC facility, supported by QMUL Research-IT. doi:10.5281/zenodo.438045.
In particular, this research made use of the OnDemand portal [@Hudak2018].


# References
