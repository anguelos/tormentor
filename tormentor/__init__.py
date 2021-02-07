r"""Segmentation aware image augmentations

This module realises basic utilities for image augmentations.

General design principles:
    - Images are always 3D tensors [CxHxW], batches are always 4D tensors [BxCxHxW]
    - Augmentations are always a applied on a single sample
    - The first dimension of the 4D tensor is always 1

"""

import types

from .augmented_dataloader import AugmentedDataLoader
from .augmented_dataset import AugmentedDs, AugmentedCocoDs
from .base_augmentation import StaticImageAugmentation, SpatialImageAugmentation, DeterministicImageAugmentation, \
    Identity, AugmentationChoice, AugmentationCascade, create_sampling_field, apply_sampling_field
from .color_augmentations import *
from .factory import AugmentationFactory
from .random import Uniform, Bernoulli, Distribution, Categorical
from .resizing_augmentation import *
from .spatial_augmentations import *
# from .backgrounds import ConstantBackground, NormalNoiseBackground, UniformNoiseBackground, PlasmaBackground
from .util import debug_pattern, render_singleline_text
from .wrap import Wrap, Shred
from .version import __version__

reset_all_seeds = DeterministicImageAugmentation.reset_all_seeds

_abstract_augmentations = {DeterministicImageAugmentation, StaticImageAugmentation, SpatialImageAugmentation,
                           ColorAugmentation, ResizingAugmentation, AugmentationCascade, AugmentationChoice}

_all_augmentations = {obj for obj in locals().values() if
                      isinstance(obj, type) and issubclass(obj, DeterministicImageAugmentation)}
_leaf_augmentations = _all_augmentations - _abstract_augmentations


def __xor__aug(self, other):
    return AugmentationFactory(self).__xor__(other)


def __pipe__aug(self, other):
    return AugmentationFactory(self).__or__(other)


for aug in _leaf_augmentations:
    aug.__xor__ = types.MethodType(__xor__aug, aug)
    aug.__or__ = types.MethodType(__pipe__aug, aug)

factory_dict = {f"Random{aug.__name__}": AugmentationFactory(aug) for aug in _leaf_augmentations}
all_factory_names = list(factory_dict.keys())
locals().update(factory_dict)

_all_augmentation_names = [aug.__name__ for aug in _all_augmentations]

__all__ = ["__version__",
           "render_singleline_text",
           "_abstract_augmentations",
           "_all_augmentations",
           "_leaf_augmentations",
           "create_sampling_field",
           "debug_pattern",
           "Constant",
           "Uniform",
           "Bernoulli",
           "Distribution",
           "Categorical",
           "AugmentationChoice",
           "Scale",
           "Rotate",
           "DeterministicImageAugmentation",
           "SpatialImageAugmentation",
           "AugmentedDs",
           "AugmentedCocoDs",
           "reset_all_seeds"] + _all_augmentation_names + all_factory_names
