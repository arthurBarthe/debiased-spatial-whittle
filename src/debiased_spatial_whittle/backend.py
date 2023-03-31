class BackendManager:
    backend_name = 'numpy'
    
    @classmethod
    def set_backend(cls, name: str):
        cls.backend_name = name
    
    @classmethod
    def get_backend(cls):
        if cls.backend_name == 'numpy':
            import numpy
            return numpy
        elif cls.backend_name == 'autograd':
            import autograd.numpy
            return autograd.numpy
        elif cls.backend_name == 'torch':
            import torch
            return torch