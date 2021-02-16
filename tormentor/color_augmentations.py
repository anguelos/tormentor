import kornia as K

from diamond_square import functional_diamond_square
from .base_augmentation import StaticImageAugmentation, AugmentationState
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
            result = type(self).functional_image(*((batch_tensor,) + state))
            return result
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
    do_inversion = Bernoulli(.5)

    def generate_batch_state(self, batch: torch.FloatTensor) -> AugmentationState:
        do_inversion = type(self).do_inversion(batch.size(0), device=batch.device).view(-1).float()
        return do_inversion,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, do_inversion: torch.FloatTensor) -> torch.FloatTensor:
        do_inversion = do_inversion.view(-1, 1, 1, 1)
        #print(do_inversion)
        hsv_batch = K.color.rgb_to_hsv(batch)
        hsv_batch[:, 2:, :, :] = (1 - hsv_batch[:, 2:, :, :]) * do_inversion + hsv_batch[:, 2:, :, :] * (1 - do_inversion)
        out_batch = K.color.hsv_to_rgb(hsv_batch)
        return out_batch


class Brightness(ColorAugmentation):
    r"""Changes the brightness of the image.

    .. image:: _static/example_images/Brightness.png
   """
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> AugmentationState:
        brightness = type(self).brightness(batch.size(0), device=batch.device).view(-1)
        return brightness,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, brightness: torch.FloatTensor) -> torch.FloatTensor:
        return K.adjust_brightness(batch, brightness)


class Saturation(ColorAugmentation):
    r"""Changes the saturation of the image.

    .. image:: _static/example_images/Saturation.png
   """
    saturation = Uniform((0.0, 2.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> AugmentationState:
        saturation = type(self).saturation(batch.size(0), device=batch.device).view(-1)
        return saturation,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, saturation: torch.FloatTensor) -> torch.FloatTensor:
        return K.adjust_saturation(batch, saturation)


class Contrast(ColorAugmentation):
    r"""Changes the contrast of the image.

    .. image:: _static/example_images/Contrast.png
   """
    contrast = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> AugmentationState:
        contrast = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        return contrast,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, contrast: torch.FloatTensor) -> torch.FloatTensor:
        # contrast = contrast.view(-1, 1, 1, 1)
        return K.adjust_saturation(batch, contrast)


class Hue(ColorAugmentation):
    r"""Changes the Hue of the image.

    .. image:: _static/example_images/Hue.png
   """
    hue = Uniform((-.5, .5))

    def generate_batch_state(self, batch: torch.FloatTensor) -> AugmentationState:
        hue = type(self).hue(batch.size(0), device=batch.device).view(-1)
        return hue,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, hue: torch.FloatTensor) -> torch.FloatTensor:
        # hue = hue.view(-1, 1, 1, 1)
        return K.adjust_hue(batch, hue)


class ColorJitter(ColorAugmentation):
    r"""Changes hue, contrast, saturation, and brightness of the image.

    .. image:: _static/example_images/ColorJitter.png
   """
    hue = Uniform((-.5, .5))
    contrast = Uniform((0.0, 1.0))
    saturation = Uniform((0.0, 2.0))
    brightness = Uniform((-1.0, 1.0))

    def generate_batch_state(self, batch: torch.FloatTensor) -> AugmentationState:
        hue = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        contrast = type(self).contrast(batch.size(0), device=batch.device).view(-1)
        saturation = type(self).saturation(batch.size(0), device=batch.device).view(-1)
        brightness = type(self).brightness(batch.size(0), device=batch.device).view(-1)
        return hue, contrast, saturation, brightness

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, hue: torch.FloatTensor, contrast: torch.FloatTensor,
                         saturation: torch.FloatTensor, brightness: torch.FloatTensor) -> torch.FloatTensor:
        batch = K.adjust_hue(batch, hue)
        batch = K.adjust_saturation(batch, saturation)
        batch = K.adjust_brightness(batch, brightness)
        batch = K.adjust_contrast(batch, contrast)
        return batch


class PlasmaBrightness(ColorAugmentation):
    r"""Changes the brightness of the image locally.

    .. image:: _static/example_images/PlasmaBrightness.png
   """
    roughness = Uniform(value_range=(.1, .7))
    intensity = Uniform(value_range=(0., 1.))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        intensity = type(self).intensity(batch_sz, device=batch_tensor.device).view(-1, 1, 1, 1)
        brightness_map = 2 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device) - 1
        return brightness_map * intensity,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, brightness_map: torch.FloatTensor) -> torch.FloatTensor:
        return torch.clamp(batch + brightness_map, 0, 1)


class PlasmaRgbBrightness(ColorAugmentation):
    r"""Changes the saturation of the image.

    .. image:: _static/example_images/Saturation.png

   """
    roughness = Uniform(value_range=(.1, .7))
    intensity = Uniform(value_range=(0., 1.))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 3, height, width)
        intensity = type(self).intensity(batch_sz, device=batch_tensor.device).view(-1, 1, 1, 1)
        brightness_map = 2 * functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device) - 1
        return brightness_map * intensity,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, brightness_map: torch.FloatTensor) -> torch.FloatTensor:
        # brightness = brightness.view(-1, 1, 1, 1)
        return torch.clamp(batch + brightness_map, 0, 1)


class PlasmaLinearColor(ColorAugmentation):
    r"""Changes the saturation of the image.

    .. image:: _static/example_images/PlasmaLinearColor.png

   """
    roughness = Uniform(value_range=(.1, .4))
    alpha_range = Uniform(value_range=(.0, 1.))
    alpha_mean = Uniform(value_range=(.0, 1.))
    beta_range = Uniform(value_range=(0., 1.))
    beta_mean = Uniform(value_range=(0., 1.))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        alpha_range = type(self).alpha_range(batch_sz, device=batch_tensor.device).view(-1,1,1,1)
        alpha_mean = type(self).alpha_mean(batch_sz, device=batch_tensor.device).view(-1,1,1,1)
        beta_range = type(self).beta_range(batch_sz, device=batch_tensor.device).view(-1,1,1,1)
        beta_mean = type(self).beta_mean(batch_sz, device=batch_tensor.device).view(-1,1,1,1)


        alpha_plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        #print("RndAlpha:",alpha_plasma.min().item(),alpha_plasma.max().item())


        alpha_plasma = (alpha_plasma * alpha_range) + (1-alpha_range)  * alpha_mean
        #print("RndAlpha2:",alpha_plasma.min().item(),alpha_plasma.max().item())

        beta_plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        #print("RndBeta:",beta_plasma.min().item(),beta_plasma.max().item())
        beta_plasma = (beta_plasma * beta_range) + (1-beta_range) * beta_mean + beta_range * .5
        #print("RndBeta2:",beta_plasma.min().item(),beta_plasma.max().item())

        #beta_available = (1 - alpha_plasma)

        #beta_alpha = beta_available * beta_alpha
        #beta_beta = (beta_available - beta_alpha) * beta_beta

        #beta_plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        #beta_plasma = beta_plasma * beta_alpha + beta_beta
        #print ("Ranges:",(alpha_plasma+beta_plasma).min().item(),(alpha_plasma+beta_plasma).max().item())
        plasma_sum = (alpha_plasma+beta_plasma)
        alpha_plasmam, beta_plasma = (alpha_plasma/plasma_sum,beta_plasma/plasma_sum)
        return alpha_plasma, beta_plasma,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, alpha_plasma: torch.FloatTensor, beta_plasma: torch.FloatTensor) -> torch.FloatTensor:
        scaled_color_img = batch * alpha_plasma + beta_plasma
        return torch.clamp(scaled_color_img, 0, 1)


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
    r"""Lowers   the brightness of the image over a random mask.

    .. image:: _static/example_images/PlasmaShadow.png
   """
    roughness = Uniform(value_range=(.1, .7))
    shade_intensity = Uniform(value_range=(-1.0, .0))
    shade_quantity = Uniform(value_range=(0.0, 1.0))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz, channels, height, width = batch_tensor.size()
        roughness = type(self).roughness(batch_sz, device=batch_tensor.device)
        shade_intensity = type(self).shade_intensity(batch_sz, device=batch_tensor.device).view(-1, 1, 1, 1)
        shade_quantity = type(self).shade_quantity(batch_sz, device=batch_tensor.device).view(-1, 1, 1, 1)
        plasma_sz = (batch_sz, 1, height, width)
        shade_map = functional_diamond_square(plasma_sz, roughness=roughness, device=batch_tensor.device)
        shade_map = (shade_map < shade_quantity).float() * shade_intensity
        return shade_map,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, shade_map: torch.FloatTensor) -> torch.FloatTensor:
        return torch.clamp(batch + shade_map, 0, 1)


class GaussianAdditiveNoise(ColorAugmentation):
    r"""Lowers   the brightness of the image over a random mask.

    .. image:: _static/example_images/PlasmaShadow.png
   """
    noise = Normal(mean=0, deviation=.2)

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        tensor_sz = batch_tensor.size()
        noise = type(self).noise(tensor_sz, device=batch_tensor.device)
        return noise,

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, noise: torch.FloatTensor) -> torch.FloatTensor:
        return torch.clamp(batch + noise, 0, 1)
