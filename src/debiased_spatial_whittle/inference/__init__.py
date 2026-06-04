from .multivariate_periodogram import Periodogram as MultivariatePeriodogram
from .periodogram import Periodogram as UnivariatePeriodogram, ExpectedPeriodogram
from .likelihood import MultivariateDebiasedWhittle, Estimator

# For backward compatibility, make Periodogram refer to MultivariatePeriodogram
Periodogram = MultivariatePeriodogram