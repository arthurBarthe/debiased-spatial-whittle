# import autograd.numpy as np
from autograd.scipy import stats
from autograd.numpy import ndarray
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.periodogram import Periodogram, ExpectedPeriodogram
from .models import CovarianceModel
from .backend import BackendManager
# TODO: why .backend??
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()

class DeWhittle:
    
    def __init__(self, z: ndarray, grid: RectangularGrid, model: CovarianceModel, use_taper:None|ndarray=None):
        self._z = z
        self.grid = grid
        
        self.use_taper = use_taper
        self.periodogram = Periodogram(taper=use_taper)
        self._I = self.periodogram(z)
        
        self.model = model
        self.free_params = model.free_params
        self.n_params = len(self.free_params)
        
    @property
    def z(self):
        return self._z
    
    @property
    def I(self):
        return self._I
        

    def expected_periodogram(self, params: ndarray) -> ndarray:
        free_params = self.model.params        
        updates = dict(zip(free_params.names, params))
        free_params.update_values(updates)
        
        ep = ExpectedPeriodogram(self.grid, self.periodogram)
        print(self.model.params)
        return ep(self.model)
    
    
    def loglik(self, params: ndarray) -> float:
        # TODO: transform params
        N = self.grid.n_points
        
        e_I = self.expected_periodogram(params)
        # TODO: may need to change constant 1/N    
        ll = -(1/N) * np.sum(np.log(e_I) + self.I / e_I)
        return ll
    
    def logprior(self, x: ndarray) -> float:
        '''uninformative prior on the transformed (unrestricted space)'''
        k = self.n_params
        return stats.multivariate_normal.logpdf(x, np.zeros(k), cov=np.eye(k)*100)
    
    
    def logpost(self, x: ndarray) -> float:
        return self.loglik(x) + self.logprior(x)
    
    # def fit(self, basinhopping:bool=False):
        





        
        
    
# M = Matrix(1000)
# print(M.inv_mat)

# class Thing:
#     def __init__(self, my_word):
#         self._word = my_word 
#     @property
#     def word(self):
#         return self._word

# print( Thing('ok').word )

# A = Thing('ok')
# A.word