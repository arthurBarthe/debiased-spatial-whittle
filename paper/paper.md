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
   index: 4
date: 20 January 2025
bibliography: paper.bib
---

# Summary
The Debiased Spatial Whittle Likelihood (DSWL) package is an open-source Python
package that implements the eponymous paper [@guillaumin_debiased_2022].
The methodology allows users to efficiently infer the parameters of stationary / homogeneous
spatial and
spatio-temporal covariance models for univariate or multivariate processes from
gridded data with potential missing observations, e.g. due to natural boundaries.
It leverages the Fast Fourier Transform, and therefore can benefit from further computational
gains through GPU implementations offered by PyTorch [@paszke2019pytorch] or
Cupy [@nishino2017cupy], both made available within
the package as alternative backends to Numpy [@harris2020array]. As such, DSWL on GPU allows to fit
covariance models to data observed on grids with tens of millions of locations.

# Statement of need
Describing patterns of spatial and spatio-temporal covariance is of interest to practitioners in
a wide range of applied sciences such as geosciences, meteorology or climate
science. Stationary covariance modelling allows for a first-order approximation
of the covariance structure, and leads to many practical applications such as
kriging [@stein] and forecasting via the conditional Gaussian multivariate
distribution. The inference of parameters for a physics-based
covariance model can also be of interest in its own right.

A major hurdle in spatio-temporal modelling is the computational cost of the
Gaussian likelihood function. This is particularly relevant for modern spatio-temporal
datasets, from physics simulations to real-world data.
The computational burden of parameter inference also arises from complex
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
Although it requires gridded data as it relies on the Fast Fourier Transform, the implemented
method additionally allows for missing observations, making it amenable to practical
applications where a full hyperrectangle of data measurements might not
be available. Missing observations might occur due to natural boundaries or
due to measurement constraints. As an example, in \autoref{fig:example} we show a simulated
sample from an exponential covariance model observed on a domain with
the shape of metropolitan France (sans Corsica), along with the distribution of estimates
obtained from 1000 independent samples generated from the same model and the
predicted distribution of estimates, which can be used to build confidence
intervals.

The package allows to address multivariate data, including those cases where the
missingness patterns might differ between the variates. For instance,
in \autoref{fig:bivariate} we show a realization of a bivariate random field
with distinct patterns of missing observations between the two variates, from which
we can still infer the parameters of the model, such as the correlation between
the two fields.
The code base also includes tapering, the use of which can further
 alleviate boundary effects [@dahlhaus_edge_1987]. Finally, the user can switch between several backends,
Numpy, Cupy and PyTorch. This allows to further benefit from computational
gains via GPU implementations of the Fast Fourier Transform.
This is shown in \autoref{fig:times} where we observed a $\times 100$
speed-up with a GPU versus a CPU.

![A simulated sample from an exponential covariance kernel observed on a domain
with the shape of metropolitan France (sans Corsica) (a),
along with the distribution of estimates obtained from 1000 independent
realizations from the same model with range parameter $\rho=14$ spatial
units (b)\label{fig:example}](france.jpeg){width=75%}

![An example of a bivariate random field with distinct patterns of missing observations
\label{fig:bivariate}](bivariate.jpg){width=75%}

![Computational time of the Debiased Spatial Whittle Likelihood averaged over
1000 samples on square grids of increasing sizes, compared between CPU and GPU (Cupy backend)
\label{fig:times}](times.jpeg){width=50%}

Other approximation techniques are available for the inference of spatio-temporal covariance
models. Among those, we can mention Vecchia-type likelihood
approximations [@katzfuss_general_2021]
such as those offered in `@jurek2023pymra; @katzfuss2023gpvecchia; @gpGp_rpackage`
and covariance tapering [@kaufman_covariance_2008],
although for the latter we are not aware of
open-source implementations.

# Software structure

The software is organized around several modules that can be grouped into the following
categories:

- grids and sampling:
  - `grids.py`: this module is used to define the rectangular grids where the data sit via the
  class RectangularGrid. A mask of zeros (missing) and ones (not missing) can be
  set to specify potential missing observations, for instance to account for natural
  boundaries
  - `simulation.py`: this module allows to efficiently sample a realization from a model on a grid
  via circulant embedding [@dietrich_fast_1997].
- models:
  - `models.py`: this module allows to define a covariance model.
    Standard covariance models are pre-defined, such as the exponential
    covariance model, the squared exponential (Gaussian) covariance model and
    the Matérn covariance model. These standard covariance models can also
    be combined (e.g. via summation) to form more complex covariance models.
- estimation:
  - `periodogram.py`: this module allows to compute the periodogram of the data, and to obtain
    the expected periodogram for a given model, grid, and periodogram combination.
  - `multivariate_periodogram.py`: this module allows to compute the periodogram for multivariate data.
  - `likelihood.py`: this module allows to define the Debiased Whittle Likelihood and the corresponding
    estimator. The optimizer can be selected among those offered by the optimize
    package of the Scipy library [@virtanen2020scipy].

A [documentation](https://debiased-spatial-whittle.readthedocs.io/en/latest/index.html)
including example notebooks is available, and issues can be raised on
[Github](https://github.com/arthurBarthe/debiased-spatial-whittle). Example notebooks can also be run directly in the browser
via [mybinder.org](https://mybinder.org/v2/gh/arthurBarthe/debiased-spatial-whittle/master).

# Acknowledgements
This research utilised Queen Mary's Apocrita HPC facility, supported by QMUL Research-IT. doi:10.5281/zenodo.438045.
In particular, this research made use of the OnDemand portal [@Hudak2018].


# References
