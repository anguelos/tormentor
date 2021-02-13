import torch

from diamond_square import functional_diamond_square
from .base_augmentation import SpatialImageAugmentation, SamplingField, AugmentationState, StaticImageAugmentation
from .random import Uniform, Bernoulli


class Wrap(SpatialImageAugmentation):
    r"""Augmentation Wrap.

    This augmentation acts like many simultaneous elastic transforms with gaussian sigmas set at varius harmonics.

    Distributions:
        ``roughness``: Quantification of the local inconsistency of the distortion effect.
        ``intensity``: Quantification of the intensity of the distortion effect.

    .. image:: _static/example_images/Wrap.png
    """
    roughness = Uniform(value_range=(.1, .7))
    intensity = Uniform(value_range=(.0, 1.))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> AugmentationState:
        batch_sz, height, width = sampling_tensors[0].size()
        roughness = type(self).roughness(batch_sz, device=sampling_tensors[0].device)
        intensity = type(self).intensity(batch_sz, device=sampling_tensors[0].device)
        plasma_sz = (batch_sz, 1, height, width)
        plasma_x = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device) - .5
        plasma_y = functional_diamond_square(plasma_sz, roughness=roughness, device=sampling_tensors[0].device) - .5
        plasma_x, plasma_y = plasma_x[:,0,:,:], plasma_y[:, 0,:,:]
        plasma_dx = plasma_x[:, :, 1:] - plasma_x[:, :, :-1]
        plasma_dy = plasma_y[:, 1:, :] - plasma_y[:, :-1, :]

        plasma_scale_x = torch.cat([abs(plasma_dx.view(batch_sz, -1).min(dim=1)[0]).view(1, -1), plasma_dx.view(batch_sz, -1).max(dim=1)[0].view(1, -1)], dim=0).max(dim=0)[0]

        plasma_scale_y = torch.cat([abs(plasma_dy.view(batch_sz, -1).min(dim=1)[0]).view(1, -1), plasma_dy.view(batch_sz, -1).max(dim=1)[0].view(1, -1)], dim=0).max(dim=0)[0]

        plasma_x /= ((plasma_scale_x.view(-1, 1, 1) * .25 * width) / intensity.view(-1, 1, 1))
        plasma_y /= ((plasma_scale_y.view(-1, 1, 1) * .25 * height) / intensity.view(-1, 1, 1))

        return plasma_x, plasma_y

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, plasma_x: torch.FloatTensor,
                                  plasma_y: torch.FloatTensor) -> SamplingField:
        field_x, field_y = sampling_field
        return field_x + plasma_x[:, :, :], field_y + plasma_y[:, :, :]


class Shred(StaticImageAugmentation):
    r"""Augmentation Shred.


    Distributions:
        ``roughness``: Quantification of the local inconsistency of the distortion effect.
        ``erase_percentile``: Quantification of the surface that will be erased.
        ``inside``: If True

    .. image:: _static/example_images/Shred.png
    """
    roughness = Uniform(value_range=(.4, .8))
    inside = Bernoulli(prob=.5)
    erase_percentile = Uniform(value_range=(.0, .5))

    def generate_batch_state(self, image_batch: torch.Tensor) -> AugmentationState:
        batch_sz, _, width, height = image_batch.size()
        roughness = type(self).roughness(batch_sz, device=image_batch.device)
        plasma_sz = (batch_sz, 1, width, height)
        plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=image_batch.device)
        inside = type(self).inside(batch_sz, device=image_batch.device).float()
        erase_percentile = type(self).erase_percentile(batch_sz, device=image_batch.device)
        return plasma, inside, erase_percentile

    @classmethod
    def functional_image(cls, image_batch: torch.Tensor, plasma: torch.FloatTensor, inside: torch.FloatTensor,
                                  erase_percentile: torch.FloatTensor) -> torch.Tensor:
        inside = inside.view(-1, 1, 1, 1)
        erase_percentile = erase_percentile.view(-1, 1, 1, 1)
        plasma = inside * plasma + (1 - inside) * (1 - plasma)
        plasma_pixels = plasma.view(plasma.size(0), -1)
        thresholds = []
        for n in range(plasma_pixels.size(0)):
            thresholds.append(torch.kthvalue(plasma_pixels[n], int(plasma_pixels.size(1) * erase_percentile[n]))[0])
        thresholds = torch.Tensor(thresholds).view(-1, 1, 1, 1).to(plasma.device)
        erase = (plasma < thresholds).float()
        return image_batch * (1 - erase)
