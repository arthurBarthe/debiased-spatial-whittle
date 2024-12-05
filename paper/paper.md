---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib
---

# Summary
Spatio-temporal stochastic processes are of interest to practitioners in
a wide variety of fields such as geosciences, meteorology or climate
science. Stationary covariance modelling allows for a first-order description
of such processes, and allows many practical applications such as
interpolation and forecasting via conditional Gaussian multivariate
distributions.

A major hurdle in spatio-temporal modelling is the computation of the
likelihood function. This is particularly relevant for modern spatio-temporal
datasets, from physics simulations to real-word data, and for complex
spatio-temporal covariance models which require a high number of likelihood
evaluation during the optimization process.

`SDW` is a Python implementation of the Spatial Debiased Whittle likelihood
[REF]. It allows to leverage the computational efficiency of the Fast
Fourier Transform to approximate the log likelihood. While the use
of the Fast Fourier Transform requires gridded data, the implemented
method allows for missing observations, making it amenable to practical
applications where a full hypercube of data measurements might not
be available. We also implement important corrections proposed in [REF]
that tackle the standard bias issue in spectral-density based estimation
methods. Finally, the code is written to allow to switch backends between
Numpy, Cupy and PyTorch. This allows to further benefits from computational
gains via GPU implementations of the FFT.

# Software description


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
