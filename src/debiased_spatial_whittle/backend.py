class BackendManager:
    backend_name = 'numpy'

    @classmethod
    def get_backend(cls):
        if cls.backend_name == 'numpy':
            import numpy
            return numpy
        elif cls.backend_name == 'torch':
            import torch
            return torch

    @classmethod
    def set_backend(cls, name):
        cls.backend_name = name