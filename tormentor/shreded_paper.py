from diamond_square import diamond_square
from .base_augmentation import SpatialImageAugmentation


class ShredBackground(SpatialImageAugmentation):
    def forward(self, tensor_image):
        batch_size, channels, width, height = tensor_image.size()
        #roughness=
        shred_ds = diamond_square(width_height=(width, height),roughness=)

    @classmethod
    def factory(cls, roughness_min=.3, roughness_max=.7, bg_mean_min=.01, bg_mean_max=.99, bg_min_dev=.0, bg_max_dev=.1,
                quantile_min=0, quantile_max=.5):
        return lambda: ShredBackground(roughness_min=roughness_min, roughness_max=roughness_max,
                                       bg_mean_min=bg_mean_min,
                                       bg_mean_max=bg_mean_max, bg_min_dev=bg_min_dev, bg_max_dev=bg_max_dev,
                                       quantile_min=quantile_min, quantile_max=quantile_max)
