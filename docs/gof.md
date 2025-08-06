Aside from the spectral-domain maximum likelihood procedure implemented for
the inference of parametric covariance models, we provide some tools for
goodness-of-fit analysis. The underlying rationale is that the user, having
fitted a parametric covariance model to some data, might want to assess
whether the model actually provides a good fit to the data.

We propose two types of residuals to carry out such an analysis,
corresponding to the two classes referenced below. Currently,
these diagnostics are only implemented for 2-dimensional
random fields.

An example notebook is made available [here](goodness_of_fit.py).
It is important to note that the diagnostics are run under the approximation
that the spectral residuals are uncorrelated. In practice, this approximation
might be a rather poor one, in particular for a covariance model
such as the squared exponential.

## diagnostics.py

::: debiased_spatial_whittle.diagnostics.GoodnessOfFitSimonsOlhede

::: debiased_spatial_whittle.diagnostics.GoodnessOfFit
