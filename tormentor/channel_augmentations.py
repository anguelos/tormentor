from .random import *
from .base_augmentation import ChannelImageAugmentation, SpatialAugmentationState
import kornia as K


class Brighten(ChannelImageAugmentation):
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch:torch.FloatTensor)->SpatialAugmentationState:
        brightness = type(self).brightness(batch.size(0)).view(-1)
        return brightness,

    @staticmethod
    def functional_image(batch:torch.FloatTensor, brightness: torch.FloatTensor)->torch.FloatTensor:
        brightness = brightness.view(-1, 1, 1, 1)
        return torch.clamp(brightness, 0.0, 1.0)


class Saturate(ChannelImageAugmentation):
    saturation = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch:torch.FloatTensor)->SpatialAugmentationState:
        saturation = type(self).saturation(batch.size(0)).view(-1)
        return saturation,

    @staticmethod
    def functional_image(batch:torch.FloatTensor, saturation: torch.FloatTensor)->torch.FloatTensor:
        saturation = saturation.view(-1, 1, 1, 1)
        return K.color.adjust_saturation(batch, saturation)


class Contrast(ChannelImageAugmentation):
    contrast = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch:torch.FloatTensor)->SpatialAugmentationState:
        contrast = type(self).contrast(batch.size(0)).view(-1)
        return contrast,

    @staticmethod
    def functional_image(batch:torch.FloatTensor, contrast: torch.FloatTensor)->torch.FloatTensor:
        contrast = contrast.view(-1, 1, 1, 1)
        return K.color.adjust_saturation(batch, contrast)
