import numpy as np
from abc import ABC, abstractmethod

from debiased_spatial_whittle.backend import BackendManager

xp = BackendManager.get_backend()


class CovarianceTaper(ABC):
    """
    Abstract base class for covariance tapers.
    
    Covariance tapers are functions that modify covariance values based on distance,
    typically used to create sparse covariance matrices for computational efficiency.
    """
    
    @abstractmethod
    def compute_taper(self, lags: xp.ndarray) -> xp.ndarray:
        """
        Compute taper values for given lags.
        
        Parameters
        ----------
        lags : xp.ndarray
            Array of shape (d, n_points) where d is the spatial dimension
            and n_points is the number of lag vectors.
            
        Returns
        -------
        xp.ndarray
            Array of shape (n_points,) containing taper values for each lag vector.
        """
        pass
    
    def __call__(self, lags: xp.ndarray) -> xp.ndarray:
        """Alias for compute_taper for convenience."""
        return self.compute_taper(lags)


class CompactCovarianceTaper(CovarianceTaper):
    """
    Compact covariance taper that becomes zero beyond a specified range.
    
    This taper implements a smooth compactly supported function that is 1 at zero lag
    and smoothly decreases to 0 at the specified range.
    
    Attributes
    ----------
    range : float
        The distance at which the taper becomes zero
    """
    
    def __init__(self, range: float = 1.0):
        self.range = float(range)
    
    def compute_taper(self, lags: xp.ndarray) -> xp.ndarray:
        """Compute compact taper values using a smooth polynomial function."""
        distances = xp.sqrt(xp.sum(lags**2, axis=0))
        normalized_dist = distances / self.range
        
        # Use a smooth polynomial taper: (1 - r^3)^3 for r <= 1, 0 otherwise
        mask = normalized_dist <= 1.0
        taper_values = xp.zeros_like(normalized_dist)
        
        # Apply smooth polynomial taper where distance <= range
        r = normalized_dist[mask]
        taper_values[mask] = (1 - r**3)**3
        
        return taper_values


class WendlandTaper(CovarianceTaper):
    """
    Wendland covariance taper function.
    
    Implements the Wendland C2 function, which is a compactly supported
    positive definite function commonly used in covariance tapering.
    
    The Wendland function is defined as:
    W(r) = (1 - r)^4 * (1 + 4r) for r <= 1, 0 otherwise
    
    Attributes
    ----------
    range : float
        The distance at which the taper becomes zero
    """
    
    def __init__(self, range: float = 1.0):
        self.range = float(range)
    
    def compute_taper(self, lags: xp.ndarray) -> xp.ndarray:
        """Compute Wendland taper values."""
        distances = xp.sqrt(xp.sum(lags**2, axis=0))
        normalized_dist = distances / self.range
        
        mask = normalized_dist <= 1.0
        taper_values = xp.zeros_like(normalized_dist)
        
        # Apply Wendland C2 function where distance <= range
        r = normalized_dist[mask]
        taper_values[mask] = (1 - r)**4 * (1 + 4 * r)
        
        return taper_values


class SphericalTaper(CovarianceTaper):
    """
    Spherical covariance taper function.
    
    Implements a simple spherical taper that decreases linearly from 1 to 0
    over the specified range.
    
    Attributes
    ----------
    range : float
        The distance at which the taper becomes zero
    """
    
    def __init__(self, range: float = 1.0):
        self.range = float(range)
    
    def compute_taper(self, lags: xp.ndarray) -> xp.ndarray:
        """Compute spherical taper values."""
        distances = xp.sqrt(xp.sum(lags**2, axis=0))
        normalized_dist = distances / self.range
        
        mask = normalized_dist <= 1.0
        taper_values = xp.zeros_like(normalized_dist)
        
        # Apply linear taper where distance <= range
        r = normalized_dist[mask]
        taper_values[mask] = 1 - r
        
        return taper_values


class ProductCovarianceTaper(CovarianceTaper):
    """
    Product covariance taper that combines two tapers, applying each to multiple dimensions.
    
    This taper is useful for anisotropic tapering where different spatial dimensions
    require different taper functions or ranges. Each sub-taper can be applied to
    one or more dimensions.
    
    Attributes
    ----------
    taper1 : CovarianceTaper
        First taper function
    
    taper2 : CovarianceTaper
        Second taper function
        
    dims1 : list[int]
        List of dimensions to apply the first taper to
        
    dims2 : list[int]
        List of dimensions to apply the second taper to
        
    Examples
    --------
    >>> from debiased_spatial_whittle.models.tapers import WendlandTaper, ProductCovarianceTaper
    >>> taper_x = WendlandTaper(range=2.0)
    >>> taper_y = WendlandTaper(range=1.0)
    >>> # Apply taper_x to dimension 0, taper_y to dimension 1
    >>> product_taper = ProductCovarianceTaper(taper_x, taper_y, dims1=[0], dims2=[1])
    >>> import numpy as np
    >>> lags = np.array([[1.0, 0.5], [0.5, 1.0]])  # 2D lags
    >>> taper_values = product_taper(lags)
    
    >>> # Apply taper_x to dimensions 0 and 1, taper_y to dimension 2 (3D case)
    >>> product_taper_3d = ProductCovarianceTaper(taper_x, taper_y, dims1=[0, 1], dims2=[2])
    """
    
    def __init__(self, taper1: CovarianceTaper, taper2: CovarianceTaper, 
                 dims1: list[int] = None, dims2: list[int] = None):
        """
        Create a product taper that applies different tapers to different sets of dimensions.
        
        Parameters
        ----------
        taper1 : CovarianceTaper
            First taper function
            
        taper2 : CovarianceTaper
            Second taper function
            
        dims1 : list[int], optional
            List of dimensions to apply first taper to. If None, applies to all dimensions.
            
        dims2 : list[int], optional
            List of dimensions to apply second taper to. If None, applies to all dimensions.
        """
        self.taper1 = taper1
        self.taper2 = taper2
        self.dims1 = dims1
        self.dims2 = dims2
    
    def compute_taper(self, lags: xp.ndarray) -> xp.ndarray:
        """
        Compute product of individual taper values.
        
        For each lag vector, computes the product of the taper values from
        each individual taper applied to their respective sets of dimensions.
        
        Parameters
        ----------
        lags : xp.ndarray
            Array of shape (d, n_points) where d is the spatial dimension
            and n_points is the number of lag vectors.
            
        Returns
        -------
        xp.ndarray
            Array of shape (n_points,) containing product taper values
        """
        # Set default dimensions if not specified
        if self.dims1 is None:
            dims1 = list(range(lags.shape[0]))
        else:
            dims1 = self.dims1
            
        if self.dims2 is None:
            dims2 = list(range(lags.shape[0]))
        else:
            dims2 = self.dims2
        
        # Extract the lag vectors for each set of dimensions
        lags1 = lags[dims1, :]
        lags2 = lags[dims2, :]
        
        # Compute taper values for each set of dimensions
        taper1_values = self.taper1.compute_taper(lags1)
        taper2_values = self.taper2.compute_taper(lags2)
        
        # Return product of taper values
        return taper1_values * taper2_values