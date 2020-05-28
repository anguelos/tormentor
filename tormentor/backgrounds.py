import torch
from diamond_square import functional_diamond_square
from .random import Uniform, Bernoulli, Categorical, Constant, Normal
from .base_augmentation import DeterministicImageAugmentation


class AbstractBackground(DeterministicImageAugmentation):
    def blend_by_mask(self, input_tensor, mask_tensor):
        res = input_tensor * mask_tensor + (1 - mask_tensor) * self.like(input_tensor)
        return res


class ConstantBackground(AbstractBackground):
    value = Constant(0)

    def forward_batch_img(self, tensor_image):
        return torch.zeros_like(tensor_image) + type(self).value()


class UniformNoiseBackground(AbstractBackground):
    pixel_values = Uniform(value_range=(0.0, 1.0))

    def forward_batch_img(self, tensor_image):
        return type(self).pixel_values(tensor_image.size())


class NormalNoiseBackground(AbstractBackground):
    pixel_values = Normal(mean=0.0, deviation=1.0)

    def forward_batch_img(self, tensor_image):
        return type(self).pixel_values(tensor_image.size())


class PlasmaBackground(AbstractBackground):
    roughness = Uniform(value_range=(0.2, 0.6))
    pixel_means = Uniform(value_range=(0.0, 1.0))
    pixel_ranges = Uniform(value_range=(0.0, 1.0))

    def forward_batch_img(self, tensor_image):
        batch_size = tensor_image.size(0)
        roughness = type(self).roughness(batch_size)
        pixel_means = type(self).pixel_means(batch_size).view(-1, 1, 1, 1)
        pixel_ranges = type(self).pixel_ranges(batch_size).view(-1, 1, 1, 1)
        plasma = functional_diamond_square(tensor_image.size(), roughness=roughness, device=self.device)
        plasma_reshaped = plasma.reshape([batch_size, -1])
        plasma_min = plasma_reshaped.min(dim=1)[0].view(-1, 1, 1, 1)
        plasma_max = plasma_reshaped.max(dim=1)[0].view(-1, 1, 1, 1)
        return (pixel_ranges * (plasma - plasma_min) / (plasma_max - plasma_min)) +  pixel_means - pixel_ranges / 2
