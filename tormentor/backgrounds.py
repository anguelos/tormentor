import torch
from diamond_square import diamond_square


class BackgroundGenerator(object):
    def create(self, size, device, dtype):
        raise NotImplementedError()

    def like(self, tensor):
        return self.create(tensor.size(), device=tensor.device, dtype=tensor.dtype)

    def blend_by_mask(self, input_tensor, mask_tensor):
        mask_tensor = mask_tensor.float()
        res = input_tensor * mask_tensor + (1 - mask_tensor) * self.like(input_tensor)
        return res


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


class PlasmaBackground(BackgroundGenerator):
    def __init__(self, mean_range, std_range, roughness_range):
        self.mean_range = mean_range
        self.std_range = std_range
        self.roughness_range = roughness_range

    def create(self, size, device, dtype=None):
        roughness = torch.rand(1).item() * (self.roughness_range[1]-self.roughness_range[0]) + self.roughness_range[0]
        mean = torch.rand(1).item() * (self.mean_range[1] - self.mean_range[0]) + self.mean_range[0]
        std = torch.rand(1).item() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]

        input_batch_size, input_channel_size, input_width, input_height = size
        ds = diamond_square(width_height=(input_width, input_height), roughness=roughness,output_deviation_mean=(std, mean), replicates=input_batch_size*input_channel_size, device=device)
        result = ds.view(size)
        if dtype is not None:
            result = result.type(dtype=dtype)
        return result
