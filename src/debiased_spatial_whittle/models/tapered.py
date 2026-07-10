from debiased_spatial_whittle.models.base import CovarianceModel, ModelParameter
from debiased_spatial_whittle.models.tapers import CovarianceTaper
from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()


class TaperedCovarianceModel(CovarianceModel):
    """
    A covariance model that applies a taper to a base covariance model.
    
    This model multiplies the covariance values from a base model by taper values
    computed from a covariance taper function. This is useful for creating sparse
    covariance matrices while preserving the overall structure of the base model.
    
    Attributes
    ----------
    base_model : CovarianceModel
        The underlying covariance model to be tapered
    
    taper : CovarianceTaper
        The taper function to apply to the covariance values
        
    range : ModelParameter
        The range parameter for the taper (passed to the taper function)
        
    Examples
    --------
    >>> from debiased_spatial_whittle.models.univariate import SquaredExponentialModel
    >>> from debiased_spatial_whittle.models.tapers import WendlandTaper
    >>> base_model = SquaredExponentialModel(rho=5.0, sigma=1.0)
    >>> taper = WendlandTaper(range=3.0)
    >>> tapered_model = TaperedCovarianceModel(base_model, taper)
    >>> import numpy as np
    >>> lags = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])  # 2D lags
    >>> tapered_model(lags)
    """
    
    range = ModelParameter(
        default=1.0, 
        bounds=(0, xp.inf), 
        doc="Range parameter for the taper function", 
        latex_display=r"r"
    )
    
    def __init__(self, base_model, taper: CovarianceTaper = None, range: float = None, name: str = None):
        """
        Initialize a tapered covariance model.
        
        Parameters
        ----------
        base_model : CovarianceModel
            The base covariance model to be tapered
        
        taper : CovarianceTaper, optional
            The taper function to use. If None, a default WendlandTaper is used.
            
        range : float, optional
            The range parameter for the taper function
            
        name : str, optional
            Name for the model
        """
        super().__init__((base_model,), range, name=name)
        
        # Set the taper function
        if taper is None:
            from debiased_spatial_whittle.models.tapers import WendlandTaper
            self._taper = WendlandTaper(range=self.range if range is None else range)
        else:
            self._taper = taper
            # Update taper range if provided
            if range is not None and hasattr(taper, 'range'):
                taper.range = range
    
    @property
    def base_model(self):
        """Get the base covariance model."""
        return self.children[0]
    
    @property
    def taper(self):
        """Get the taper function."""
        return self._taper
    
    @taper.setter
    def taper(self, value):
        """Set the taper function."""
        self._taper = value
        # Update taper range to match model range if applicable
        if hasattr(value, 'range') and hasattr(self, '_range'):
            value.range = self.range
    
    def compute(self, lags: xp.ndarray, range: xp.ndarray, *params) -> xp.ndarray:
        """
        Compute the tapered covariance.
        
        Parameters
        ----------
        lags : xp.ndarray
            Array of shape (d, n_points) where d is spatial dimension
            
        range : xp.ndarray
            Range parameter for the taper
            
        params : tuple
            Parameters for the base model
            
        Returns
        -------
        xp.ndarray
            Array of shape (n_points,) containing tapered covariance values
        """
        # Update taper range
        self.taper.range = float(range.item() if hasattr(range, 'item') else range)
        
        # Compute base model covariance (pass only the base model parameters, not the range)
        base_cov = self.children[0].compute(lags, *params)
        
        # Compute taper values
        taper_values = self.taper.compute_taper(lags)
        
        # Apply taper to covariance
        return base_cov * taper_values