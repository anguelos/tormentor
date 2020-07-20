from .random import *
from .base_augmentation import StaticImageAugmentation, SpatialAugmentationState
from diamond_square import functional_diamond_square
import kornia as K

class ColorAugmentation(StaticImageAugmentation):
    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        state = self.generate_batch_state(batch_tensor)
        if batch_tensor.size(1) == 3: # Color operations require
            return type(self).functional_image(*((batch_tensor,) + state))
        if batch_tensor.size(1) == 1: # Color operations require
            batch_tensor = batch_tensor.repeat([1,3,1,1])
            batch_tensor = type(self).functional_image(*((batch_tensor,) + state))
            return K.color.rgb_to_grayscale(batch_tensor)
        else: # No colors were in the image, it will be ignored
            return batch_tensor


class BrightnessChanger(ColorAugmentation):
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch:torch.FloatTensor)->SpatialAugmentationState:
        brightness = type(self).brightness(batch.size(0), device=batch.device).view(-1)
        return brightness,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, brightness: torch.FloatTensor)->torch.FloatTensor:
        #brightness = brightness.view(-1, 1, 1, 1)
        return K.color.adjust_brightness(batch, brightness)


class SaturationChanger(ColorAugmentation):
    saturation = Uniform((0.0, 2.0))

    def generate_batch_state(self, batch:torch.FloatTensor)->SpatialAugmentationState:
        saturation = type(self).saturation(batch.size(0), device=batch.device).view(-1)
        return saturation,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, saturation: torch.FloatTensor)->torch.FloatTensor:
        #saturation = saturation.view(-1, 1, 1, 1)
        return K.color.adjust_saturation(batch, saturation)


class ContrastChanger(ColorAugmentation):
    contrast = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor)->SpatialAugmentationState:
        contrast = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        return contrast,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, contrast: torch.FloatTensor)->torch.FloatTensor:
        #contrast = contrast.view(-1, 1, 1, 1)
        return K.color.adjust_saturation(batch, contrast)

class HueChanger(ColorAugmentation):
    hue = Uniform((-.5, .5))

    def generate_batch_state(self, batch:torch.FloatTensor) -> SpatialAugmentationState:
        hue = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        return hue,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, hue: torch.FloatTensor) -> torch.FloatTensor:
        #hue = hue.view(-1, 1, 1, 1)
        return K.color.adjust_hue(batch, hue)

class ColorJitter(ColorAugmentation):
    hue = Uniform((-.5, .5))
    contrast = Uniform((0.0, 1.0))
    saturation = Uniform((0.0, 2.0))
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch:torch.FloatTensor) -> SpatialAugmentationState:
        hue = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        contrast = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        saturation = type(self).saturation(batch.size(0), device=batch.device).view(-1)
        brightness = type(self).brightness(batch.size(0), device=batch.device).view(-1)
        return hue, contrast, saturation, brightness

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, hue: torch.FloatTensor, contrast: torch.FloatTensor, saturation: torch.FloatTensor, brightness: torch.FloatTensor) -> torch.FloatTensor:
        batch = K.color.adjust_hue(batch, hue)
        batch = K.color.adjust_saturation(batch, saturation)
        batch = K.color.adjust_brightness(batch, brightness)
        batch = K.color.adjust_contrast(batch, contrast)
        return batch



class PlasmaBrightness(ColorAugmentation):
    roughness = Uniform(value_range=(.1, .7))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        brightness_map = 2 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device) - 1
        return brightness_map,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, brightness_map: torch.FloatTensor)->torch.FloatTensor:
        #brightness = brightness.view(-1, 1, 1, 1)
        return torch.clamp(batch + brightness_map, 0, 1)


class PlasmaContrast(ColorAugmentation):
    roughness = Uniform(value_range=(.1, .7))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        contrast_map = 4 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        return contrast_map,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, contrast_map: torch.FloatTensor)->torch.FloatTensor:
        return torch.clamp((batch - .5) * contrast_map + .5, 0, 1)


class PlasmaShadow(ColorAugmentation):
    roughness = Uniform(value_range=(.1, .7))
    shade_intencity = Uniform(value_range=(-1.0, .0))
    shade_quantity = Uniform(value_range=(0.0, 1.0))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        shade_intencity = type(self).shade_intencity(batch_sz, device=batch_tensor.device).view(-1, 1, 1)
        shade_quantity = type(self).shade_quantity(batch_sz, device=batch_tensor.device).view(-1, 1, 1)
        plasma_sz = (batch_sz, 1, height, width)
        shade_map = functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        shade_map = (shade_map > shade_quantity).float() * shade_intencity
        return shade_map,

    @classmethod
    def functional_image(cls, batch:torch.FloatTensor, shade_map: torch.FloatTensor)->torch.FloatTensor:
        return torch.clamp(batch + shade_map, 0, 1)
