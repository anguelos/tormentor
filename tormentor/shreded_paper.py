from diamond_square import diamond_square
from .base_augmentation import SpatialImageAugmentation
import torch


class Shred(SpatialImageAugmentation):
    def forward(self, tensor_image):
        batch_size, channels, width, height = tensor_image.size()
        roughness = torch.rand(1).item()*(self.roughness_max-self.roughness_min)+self.roughness_min
        quantile = torch.rand(1).item() * (self.quantile_max - self.quantile_min) + self.quantile_min

        mask_ds = diamond_square(width_height=(width, height), roughness=roughness)
        hole = torch.rand(1).item() > self.hole_prob
        if hole:
            mask = mask_ds > mask_ds.view(-1).kthvalue(int(mask_ds.view(-1).size() * quantile))[0]
        else:
            mask = mask_ds < mask_ds.view(-1).kthvalue(int(mask_ds.view(-1).size() * (1-quantile)))[0]
        mask = mask.float()
        res = mask * tensor_image
        return res



    @classmethod
    def factory(cls, roughness_min=.3, roughness_max=.7, bg_mean_min=.01, bg_mean_max=.99, bg_min_dev=.0, bg_max_dev=.1,
                quantile_min=0, quantile_max=.5, hole_prob = .5):
        return lambda: Shred(roughness_min=roughness_min, roughness_max=roughness_max,
                             bg_mean_min=bg_mean_min,
                             bg_mean_max=bg_mean_max, bg_min_dev=bg_min_dev, bg_max_dev=bg_max_dev,
                             quantile_min=quantile_min, quantile_max=quantile_max, hole_prob=hole_prob)
