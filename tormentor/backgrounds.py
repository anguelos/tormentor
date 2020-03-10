import torch


class BackgroundGenerator(object):
    def create(self, size, device, dtype):
        raise NotImplementedError()

    def like(self, tensor):
        return self.create(tensor.size(), device=tensor.device, dtype=tensor.dtype)


class ConstantBackground(BackgroundGenerator):
    def __init__(self, value=0):
        self.value = value

    def create(self, size, device, dtype):
        result = torch.zeros(size, device=device, dtype=dtype) + self.value
        return result


class UniformNoiseBackground(BackgroundGenerator):
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.value_range = max_value - min_value

    def create(self, size, device, dtype):
        return torch.rand(size, device=device, dtype=dtype) * self.value_range + self.min_value


class GaussianNoiseBackground(BackgroundGenerator):
    def __init__(self, mean=0.0, variance=1.0):
        self.mean = mean
        self.variance = variance

    def create(self, size, device, dtype):
        return torch.rand(size, device=device, dtype=dtype) * self.variance + self.mean

# TODO(anguelos) Implement Image backgrounds
# TODO(anguelos) Implement Plasma backgrounds
