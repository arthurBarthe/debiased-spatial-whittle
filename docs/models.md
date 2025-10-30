
## ::: debiased_spatial_whittle.models
    options:
      members: false

This package provides tools to define covariance models. A few pre-defined covariance
models are made available. To define your own covariance model, you need
to inherit from CovarianceModel or CompoundCovarianceModel.
In both cases, you need to define the _compute method, which will expect
a lag array with shape (ndim, n1, ..., nk, 1) where ndim is the number
of spatio-temporal dimensions, n1, ..., nk, can have any shape.
The trailing 1 is not needed in calls to the model as it is automatically added by
the inherited __call__ method.
More specifically, the trailing dimension is used to adress the case where
the user sets each parameter to be array-valued rather than scalar. This allows
parallel evaluation of the likelihood for several model parameters. This feature
can be ignored by new users of the package.

Within the _compute method of a CompoundModel, when evaluating children models at lags,
you should not use their call method but rather their _compute method (see e.g. SumModel).

For a univariate model, if we pass a lags array with shape
(ndim, n1, ..., nk), we expect the covariance model to return an array with
shape (n1, ..., nk).

For a bivariate model, we expect the covariance model to return an array with
shape (n1, ..., nk, 2, 2).

### ::: debiased_spatial_whittle.models.base

### ::: debiased_spatial_whittle.models.univariate

### ::: debiased_spatial_whittle.models.bivariate

### ::: debiased_spatial_whittle.models.spectral
