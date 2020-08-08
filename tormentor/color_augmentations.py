import kornia as K

from diamond_square import functional_diamond_square
from .base_augmentation import StaticImageAugmentation, SpatialAugmentationState
from .random import *


class ColorAugmentation(StaticImageAugmentation):
    r"""Abstract class for all augmentations manipulating the colorspace.

    All augmentations inheriting ``ColorAugmentation``, expect 3-channel inputs that can be interpreted as RGB in the
    range [0., 1.]. If the channels are neither 3 or 1, the augmentation becomes an identity.
    The subclasses should only define ``generate_batch_state(self, batch: torch.FloatTensor)`` and classmethod
    ``functional_image(cls, batch: torch.FloatTensor, *batch_state)``.
    """
    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        state = self.generate_batch_state(batch_tensor)
        if batch_tensor.size(1) == 3:  # Color operations require
            return type(self).functional_image(*((batch_tensor,) + state))
        if batch_tensor.size(1) == 1:  # Color operations require
            batch_tensor = batch_tensor.repeat([1, 3, 1, 1])
            batch_tensor = type(self).functional_image(*((batch_tensor,) + state))
            return K.color.rgb_to_grayscale(batch_tensor)
        else:  # No colors were in the image, it will be ignored
            return batch_tensor


class Invert(ColorAugmentation):
    r"""Performs color inversion in HSV colorspace for some images randomly selected.

    .. image:: _static/example_images/Invert.png
   """
    do_inversion = Bernoulli(.2)

    def generate_batch_state(self, batch: torch.FloatTensor) -> SpatialAugmentationState:
        do_inversion = type(self).do_inversion(batch.size(0), device=batch.device).view(-1).float()
        return do_inversion,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, do_inversion: torch.FloatTensor) -> torch.FloatTensor:
        do_inversion = do_inversion.view(-1, 1, 1)
        hsv_batch = K.color.rgb_to_hsv(batch)
        hsv_batch[:, 0, :, :] = (1 - hsv_batch[:, 0, :, :]) * do_inversion + hsv_batch[:, 0, :, :] * (1 - do_inversion)
        return K.color.hsv_to_rgb(hsv_batch)


class Brightness(ColorAugmentation):
    r"""Changes the brightness of the image.

    .. image:: _static/example_images/Brightness.png
   """
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> SpatialAugmentationState:
        brightness = type(self).brightness(batch.size(0), device=batch.device).view(-1)
        return brightness,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, brightness: torch.FloatTensor) -> torch.FloatTensor:
        return K.color.adjust_brightness(batch, brightness)


class Saturation(ColorAugmentation):
    r"""Changes the saturation of the image.

    .. image:: _static/example_images/Saturation.png
   """
    saturation = Uniform((0.0, 2.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> SpatialAugmentationState:
        saturation = type(self).saturation(batch.size(0), device=batch.device).view(-1)
        return saturation,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, saturation: torch.FloatTensor) -> torch.FloatTensor:
        return K.color.adjust_saturation(batch, saturation)


class Contrast(ColorAugmentation):
    r"""Changes the contrast of the image.

    .. image:: _static/example_images/Contrast.png
   """
    contrast = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> SpatialAugmentationState:
        contrast = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        return contrast,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, contrast: torch.FloatTensor) -> torch.FloatTensor:
        # contrast = contrast.view(-1, 1, 1, 1)
        return K.color.adjust_saturation(batch, contrast)


class Hue(ColorAugmentation):
    r"""Changes the Hue of the image.

    .. image:: _static/example_images/Hue.png
   """
    hue = Uniform((-.5, .5))

    def generate_batch_state(self, batch: torch.FloatTensor) -> SpatialAugmentationState:
        hue = type(self).hue(batch.size(0), device=batch.device).view(-1)
        return hue,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, hue: torch.FloatTensor) -> torch.FloatTensor:
        # hue = hue.view(-1, 1, 1, 1)
        return K.color.adjust_hue(batch, hue)


class ColorJitter(ColorAugmentation):
    r"""Changes hue, contrast, saturation, and brightness of the image.

    .. image:: _static/example_images/ColorJitter.png
   """
    hue = Uniform((-.5, .5))
    contrast = Uniform((0.0, 1.0))
    saturation = Uniform((0.0, 2.0))
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> SpatialAugmentationState:
        hue = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        contrast = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        saturation = type(self).saturation(batch.size(0), device=batch.device).view(-1)
        brightness = type(self).brightness(batch.size(0), device=batch.device).view(-1)
        return hue, contrast, saturation, brightness

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, hue: torch.FloatTensor, contrast: torch.FloatTensor,
                         saturation: torch.FloatTensor, brightness: torch.FloatTensor) -> torch.FloatTensor:
        batch = K.color.adjust_hue(batch, hue)
        batch = K.color.adjust_saturation(batch, saturation)
        batch = K.color.adjust_brightness(batch, brightness)
        batch = K.color.adjust_contrast(batch, contrast)
        return batch


class PlasmaBrightness(ColorAugmentation):
    r"""Changes the brightness of the image locally.

    .. image:: _static/example_images/PlasmaBrightness.png
   """
    roughness = Uniform(value_range=(.1, .7))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        brightness_map = 2 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device) - 1
        return brightness_map,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, brightness_map: torch.FloatTensor) -> torch.FloatTensor:
        # brightness = brightness.view(-1, 1, 1, 1)
        return torch.clamp(batch + brightness_map, 0, 1)


class PlasmaRgbBrightness(ColorAugmentation):
    r"""Changes the saturation of the image.

    .. image:: _static/example_images/Saturation.png

   """
    roughness = Uniform(value_range=(.1, .7))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 3, height, width)
        brightness_map = 2 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device) - 1
        return brightness_map,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, brightness_map: torch.FloatTensor) -> torch.FloatTensor:
        # brightness = brightness.view(-1, 1, 1, 1)
        return torch.clamp(batch + brightness_map, 0, 1)


class PlasmaContrast(ColorAugmentation):
    r"""Changes the contrast of the image locally.

    .. image:: _static/example_images/PlasmaContrast.png

   """
    roughness = Uniform(value_range=(.1, .7))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        contrast_map = 4 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        return contrast_map,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, contrast_map: torch.FloatTensor) -> torch.FloatTensor:
        return torch.clamp((batch - .5) * contrast_map + .5, 0, 1)


class PlasmaShadow(ColorAugmentation):
    r"""Lowers the brightness of the image over a random mask.

    .. image:: _static/example_images/PlasmaShadow.png

   """
    roughness = Uniform(value_range=(.1, .7))
    shade_intencity = Uniform(value_range=(-1.0, .0))
    shade_quantity = Uniform(value_range=(0.0, 1.0))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        shade_intencity = type(self).shade_intencity(batch_sz, device=batch_tensor.device).view(-1, 1, 1, 1)
        shade_quantity = type(self).shade_quantity(batch_sz, device=batch_tensor.device).view(-1, 1, 1, 1)
        plasma_sz = (batch_sz, 1, height, width)
        shade_map = functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        shade_map = (shade_map < shade_quantity).float() * shade_intencity
        return shade_map,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, shade_map: torch.FloatTensor) -> torch.FloatTensor:
        return torch.clamp(batch + shade_map, 0, 1)
