The Debiased Whittle Likelihood estimator is based on the periodogram
of the data and the expected periodogram corresponding to the combination
of a sampling grid and covariance model.

## periodogram.py
This module provides tools to define a periodogram and to obtain
its expectation, the expected periodogram.

The periodogram is defined according to,
$$
I(\mathbf{k}) =
\frac{1}{|n|}
\left|
    \sum_{\mathbf{s}}
    {
        g_{\mathbf{s}}
        X_{\mathbf{s}}
        \exp(-i\mathbf{k}\cdot\mathbf{s})
    }
\right|^2
$$
where the summation is over grid points, and where $g_{\mathbf{s}}$
is obtained from both the grid mask and the periodogram taper function.
In practice, this is implemented using the Fast Fourier Transform.

::: debiased_spatial_whittle.periodogram.Periodogram

::: debiased_spatial_whittle.periodogram.ExpectedPeriodogram


## multivariate_periodogram.py

This module provides a class for the definition of a multi-variate periodogram.

::: debiased_spatial_whittle.multivariate_periodogram.Periodogram


## likelihood.py
This module provides tools to define the Debiased Whittle Likelihood
and the associated estimator. More specifically, the Debiased Whittle Likelihood
is defined by,
$$
l(\boldsymbol{\theta})
=
\sum_{\mathbf{k}\in\Omega}
\left[
    \log\overline{I}(\mathbf{k}; \boldsymbol{\theta})
    +
    \frac
    {
        I(\mathbf{k})
    }
    {
        \overline{I}(\mathbf{k}; \boldsymbol{\theta})
    }
\right]
$$
where $I(\mathbf{k})$ is the periodogram at space-time frequency $\mathbf{k}$,
$\overline{I}(\mathbf{k}; \boldsymbol{\theta})$ is the expected periodogram and
$\Omega$ is the set of Fourier frequency (one can choose to select a subset of frequencies though).

The Debiased Whittle Likelihood Estimator then uses numerical optimization to approximately solve,

$$
\widehat{\boldsymbol{\theta}} = \argmin_{\boldsymbol{\theta}\in\Theta} l(\boldsymbol{\theta}),
$$
where $\Theta$ is the parameter space, defined within the specified model.

::: debiased_spatial_whittle.likelihood.DebiasedWhittle

::: debiased_spatial_whittle.likelihood.MultivariateDebiasedWhittle

::: debiased_spatial_whittle.likelihood.Estimator

## least_squares.py

This module provides a method for a least-squares fit of the expected periodogram
to the periodogram. This can be used for instance to obtain an initial guess
before using Debiased Whittle estimation.

::: debiased_spatial_whittle.least_squares.LeastSquareEstimator
