import kornia
import torch

from diamond_square import diamond_square
from .base_augmentation import SpatialImageAugmentation, aug_parameters

@aug_parameters(min_roughness=.3, max_roughness=.8, min_pixel_scale=0.0, max_pixel_scale=1.0)
class WrapAugmentation(SpatialImageAugmentation):
    def forward_sample(self, input_image):
        width, height = input_image.size(2), input_image.size(3)
        roughness = torch.rand(1) * (self.max_roughness - self.min_roughness) + self.min_roughness
        pixel_scale = torch.rand(1) * (self.max_pixel_scale - self.min_pixel_scale) + self.min_pixel_scale
        ds = diamond_square(width_height=(width, height), roughness=roughness, replicates=2, output_range=[-.5, .5])
        grid = kornia.utils.create_meshgrid(width, height, False)
        delta_y = ds[:, :, :, 1:] - ds[:, :, :, :-1]
        delta_x = ds[:, :, 1:, :] - ds[:, :, :-1, :]
        delta = torch.cat([delta_x.view(-1), delta_y.view(-1)], dim=0).abs()
        ds = pixel_scale * ds / delta.max() + pixel_scale/2
        result_image = kornia.geometry.remap(input_image, grid[:, :, :, 0] + ds[0, :, :, :],
                                             grid[:, :, :, 1] + ds[1, :, :, :])
        return result_image
