import torch


class BackgroundGenerator(object):
    def create(self, size, device):
        pass

    def like(self, tensor):
        raise NotImplementedError()


class ConstantBackground(BackgroundGenerator):
    def __init__(self, value=0):
        self.value = value

    def like(self, tensor):
        return torch.zeros_like(tensor) + self.value


class UniformNoiseBackground(BackgroundGenerator):
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.value_range = max_value - min_value

    def like(self, tensor):
        return torch.rand_like(tensor) * self.value_range + self.min_value
