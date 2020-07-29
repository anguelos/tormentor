r"""Segmentation aware image augmentations

This module realises basic utilities for image augmentations.

General design principles:
    - Images are always 3D tensors [CxHxW], batches are always 4D tensors [BxCxHxW]
    - Augmentations are always a applied on a single sample
    - The first dimension of the 4D tensor is always 1

"""

import types
from .random import Uniform, Bernoulli, Distribution, Categorical
from .base_augmentation import StaticImageAugmentation, SpatialImageAugmentation, DeterministicImageAugmentation, AugmentationChoice, AugmentationCascade
from .factory import AugmentationFactory
from .spatial_augmentations import *
from .color_augmentations import *
from .augmented_dataset import AugmentedDs, AugmentedCocoDs
from .augmented_dataloader import AugmentedDataLoader
from .wrap import Wrap, Shred
from .backgrounds import ConstantBackground, NormalNoiseBackground, UniformNoiseBackground, PlasmaBackground
from .util import debug_pattern

reset_all_seeds = DeterministicImageAugmentation.reset_all_seeds

leaf_augmentations = tuple(SpatialImageAugmentation.__subclasses__()) + tuple(StaticImageAugmentation.__subclasses__()) + tuple(ColorAugmentation.__subclasses__())

all_factory_names = []
all_augmentation_names = []

def __or__(self, other):
    return AugmentationFactory(self).__or__(other)

def __and__(self, other):
    return AugmentationFactory(self).__and__(other)


for aug in leaf_augmentations:
    aug.__or__ = types.MethodType(__or__, aug)
    aug.__and__ = types.MethodType(__and__, aug)
factory_dict = {f"Random{aug.__name__}": AugmentationFactory(aug) for aug in leaf_augmentations}
all_factory_names = list(factory_dict.keys())
locals().update(factory_dict)


__all__ = [
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
    "reset_all_seeds"   ] + all_augmentation_names + all_factory_names
