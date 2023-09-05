class BackendManager:
    backend_name = 'numpy'
    backend = None

    @classmethod
    def get_backend(cls):
        if cls.backend_name == 'numpy':
            import numpy
            cls.backend = numpy
            return numpy
        elif cls.backend_name == 'torch':
            import torch
            cls.backend = torch
            torch.array = torch.tensor
            torch.ndarray = torch.Tensor
            torch.concatenate = torch.cat
            torch.expand_dims = torch.unsqueeze
            torch.take = lambda x, y, z: torch.index_select(x, z, y)
            torch.transpose = lambda x, y: torch.transpose(x, -1, -2)
            return torch

    @classmethod
    def import_function(cls, *func_names):
        return [getattr(cls.backend, func_name) for func_name in func_names]

    @classmethod
    def set_backend(cls, name):
        print(f'Backend set to {name}')
        cls.backend_name = name