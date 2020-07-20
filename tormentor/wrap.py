import torch

from diamond_square import functional_diamond_square
from .base_augmentation import SpatialImageAugmentation, SamplingField, SpatialAugmentationState
from .random import Uniform, Bernoulli


class Wrap(SpatialImageAugmentation):
    roughness = Uniform(value_range=(.1, .7))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> SpatialAugmentationState:
        batch_sz, height, width = sampling_tensors[0].size()
        roughness = type(self).roughness(batch_sz, device=sampling_tensors[0].device)
        plasma_sz = (batch_sz, 1, height, width)
        plasma_x = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device) - .5
        plasma_y = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device) - .5
        plasma_x, plasma_y = plasma_x[:,0,:,:], plasma_y[:, 0,:,:]
        plasma_dx = plasma_x[:, :, 1:] - plasma_x[:, :, :-1]
        plasma_dy = plasma_y[:, 1:, :] - plasma_y[:, :-1, :]
        plasma_scale_x = torch.cat([abs(plasma_dx.view(batch_sz, -1).min(dim=1)[0]).view(1, -1),
                                    plasma_dx.view(batch_sz, -1).max(dim=1)[0].view(1, -1)], dim=0).max(dim=0)[0]

        plasma_scale_y = torch.cat([abs(plasma_dy.view(batch_sz, -1).min(dim=1)[0]).view(1, -1),
                                    plasma_dy.view(batch_sz, -1).max(dim=1)[0].view(1, -1)], dim=0).max(dim=0)[0]

        plasma_x /= (plasma_scale_x.view(-1, 1, 1) * .25 * width)
        plasma_y /= (plasma_scale_y.view(-1, 1, 1) * .25 * height)
        return plasma_x, plasma_y

    @staticmethod
    def functional_sampling_field(sampling_field: SamplingField, plasma_x: torch.FloatTensor,
                                  plasma_y: torch.FloatTensor) -> SamplingField:
        field_x, field_y = sampling_field
        return field_x + plasma_x[:, :, :], field_y + plasma_y[:, :, :]


class ShredAugmentation(SpatialImageAugmentation):
    roughness = Uniform(value_range=(.4, .8))
    inside = Bernoulli(prob=.5)
    erase_percentile = Uniform(value_range=(.0, .5))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> SpatialAugmentationState:
        batch_sz, width, height = sampling_tensors[0].size()
        roughness = type(self).roughness(batch_sz, device=sampling_tensors[0].device)
        plasma_sz = (batch_sz, 1, width, height)
        plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device)
        inside = type(self).pixel_scale(batch_sz, device=sampling_tensors[0].device).float()
        erase_percentile = self.erase_percentile(batch_sz)
        return plasma, inside, erase_percentile

    @staticmethod
    def functional_sampling_field(sampling_field: SamplingField, plasma: torch.FloatTensor, inside: torch.FloatTensor,
                                  erase_percentile: torch.FloatTensor) -> SamplingField:
        inside = inside.view(-1, 1, 1, 1)
        erase_percentile = erase_percentile.view(-1, 1, 1, 1)
        plasma = inside * plasma + (1 - inside) * (1 - plasma)
        plasma_pixels = plasma.view(plasma.size(0), -1)
        thresholds = []
        for n in range(plasma_pixels.size(0)):
            thresholds.append(torch.kthvalue(plasma_pixels[n], int(plasma_pixels.size(1) * erase_percentile))[0])
        thresholds = torch.Tensor(thresholds).view(-1, 1, 1, 1)
        erase = (plasma < thresholds) * ShredAugmentation.outside_field
        return sampling_field[0] + erase, sampling_field[1] + erase
