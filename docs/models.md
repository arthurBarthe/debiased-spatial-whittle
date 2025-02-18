
## models.py
This module provides tools to define covariance models. A few pre-defined covariance
models are made available. To define your own covariance model, you need
to inherit from CovarianceModel or CompoundCovarianceModel.
In both cases, you need to define the _compute method, which will expect
a lag array with shape (ndim, n1, ..., nk, 1) where ndim is the number
of spatio-temporal dimensions, n1, ..., nk, can have any shape.
The trailing 1 is not needed in calls to the model as it is automatically added.
Within the _compute method of a CompoundModel, when evaluating children models at lags,
you should not use their call method but rather their _compute method (see e.g. SumModel).

::: debiased_spatial_whittle.models.CovarianceModel

::: debiased_spatial_whittle.models.ExponentialModel
    options:
      show_inheritance_diagram: True
      members:
        - __call__
        - _gradient

::: debiased_spatial_whittle.models.SquaredExponentialModel

::: debiased_spatial_whittle.models.BivariateUniformCorrelation
