from .base_augmentation import SpatialImageAugmentation, SamplingField, SpatialAugmentationState
from .random import Uniform, Bernoulli
from diamond_square import functional_diamond_square
import torch


class WrapAugmentation(SpatialImageAugmentation):
    pixel_scale = Uniform(value_range=(1.0, 10.0))
    roughness = Uniform(value_range=(.4, .8))

    def generate_batch_state(self, sampling_tensors:SamplingField)->SpatialAugmentationState:
        batch_sz, width, height = sampling_tensors[0].size()
        pixel_scales = type(self).pixel_scale(batch_sz, device=sampling_tensors[0].device)
        roughness = type(self).roughness(batch_sz, device=sampling_tensors[0].device)
        plasma_sz = (batch_sz, 1, width, height)
        plasma_x = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device)
        plasma_y = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device)
        return plasma_x, plasma_y, pixel_scales

    @staticmethod
    def functional_points(sampling_field:SamplingField, plasma_x:torch.FloatTensor, plasma_y:torch.FloatTensor, pixel_scales:torch.FloatTensor)->SamplingField:
        pixel_scales = pixel_scales.view(-1, 1, 1, 1)
        field_x, field_y = sampling_field
        return field_x + plasma_x * pixel_scales, field_y + plasma_y * pixel_scales


class Shred(SpatialImageAugmentation):
    roughness = Uniform(value_range=(.4, .8))
    inside = Bernoulli(prob=.5)
    erase_percentile = Uniform(value_range=(.0, .5))

    def generate_batch_state(self, sampling_tensors:SamplingField)->SpatialAugmentationState:
        batch_sz, width, height = sampling_tensors[0].size()
        roughness = type(self).roughness(batch_sz, device=sampling_tensors[0].device)
        plasma_sz = (batch_sz, 1, width, height)
        plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device)
        inside = type(self).pixel_scale(batch_sz, device=sampling_tensors[0].device).float()
        erase_percentile = self.erase_percentile(batch_sz)
        return plasma, inside, erase_percentile

    @staticmethod
    def functional_points(sampling_field:SamplingField, plasma:torch.FloatTensor, inside:torch.FloatTensor, erase_percentile:torch.FloatTensor)->SamplingField:
        inside = inside.view(-1, 1, 1, 1)
        erase_percentile = erase_percentile.view(-1, 1, 1, 1)
        plasma = inside * plasma + (1-inside) * (1-plasma)
        plasma_pixels = plasma.view(plasma.size(0),-1)
        thresholds = []
        for n in range(plasma_pixels.size(0)):
            thresholds.append(torch.kthvalue(plasma_pixels[n], int(plasma_pixels.size(1)*erase_percentile))[0])
        thresholds = torch.Tensor(thresholds).view(-1, 1, 1, 1)
        erase = (plasma < thresholds) * Shred.outside_field
        return sampling_field[0] + erase, sampling_field[1] + erase
