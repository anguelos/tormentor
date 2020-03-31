import math
import kornia as K
from.random import *
from .base_augmentation import ChannelImageAugmentation, aug_distributions


@aug_distributions(brightness=Uniform((-1.0, 1.0)))
class Brighten(ChannelImageAugmentation):
    def forward_batch_img(self, batch_tensor):
        return K.color.adjust_brightness(batch_tensor, self.brightness(batch_tensor.size(0)))


@aug_distributions(saturation=Uniform((0.0, 1.0)))
class Saturate(ChannelImageAugmentation):
    def forward_batch_img(self, batch_tensor):
        return K.color.adjust_saturation(batch_tensor, self.saturation(batch_tensor.size(0)))


@aug_distributions(contrast=Uniform((0.0, 1.0)))
class SetContrast(ChannelImageAugmentation):
    def forward_batch_img(self, batch_tensor):
        return K.color.adjust_contrast(batch_tensor, self.contrast(batch_tensor.size(0)))


@aug_distributions(gamma=Uniform((0.0, 1.0)), flip=Bernoulli(.5))
class SetGamma(ChannelImageAugmentation):
    def forward_batch_img(self, batch_tensor):
        flip = self.flip(batch_tensor.size(0)).float()
        gamma = self.saturation(batch_tensor.size(0))
        gamma = (1 / gamma) * flip + gamma * (1 - flip)
        return K.color.adjust_gamma(batch_tensor, gamma)


@aug_distributions(gamma=Uniform((-math.pi, math.pi)), flip=Bernoulli(.5))
class SetGamma(ChannelImageAugmentation):
    def forward_batch_img(self, batch_tensor):
        flip = self.flip(batch_tensor.size(0)).float()
        gamma = self.saturation(batch_tensor.size(0))
        gamma = (1 / gamma) * flip + gamma * (1 - flip)
        return K.color.adjust_gamma(batch_tensor, gamma)

