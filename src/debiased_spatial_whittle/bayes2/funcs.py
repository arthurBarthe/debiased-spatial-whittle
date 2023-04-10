from debiased_spatial_whittle.backend import BackendManager
BackendManager.set_backend('autograd')
np = BackendManager.get_backend()


from autograd import grad
from debiased_spatial_whittle.bayes import f



gradf = grad(f)


