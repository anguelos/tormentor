r"""Segmentation aware image augmentations

This module realises basic utilities for image augmentations.

Augmentations are either spatial or

General design principles:
    - Images are always 4D tensors
    - Augmentations are always a applied on a single sample
    - The first dimension of the 4D tensor is always 1

"""
from .random import Uniform, Uniform2D, Bernoulli, Distribution, Distribution2D, Categorical
from .base_augmentation import ChannelImageAugmentation, SpatialImageAugmentation, DeterministicImageAugmentation
from .spatial_augmentations import *
from .channel_augmentations import *
from .augmented_dataset import ImageAugmentationPipelineDataset
from .wrap import WrapAugmentation
from .augmented_dataset import AugmentationDataset, ImageAugmentationPipelineDataset
reset_all_seeds = DeterministicImageAugmentation.reset_all_seeds

leaf_augmentations = tuple(SpatialImageAugmentation.__subclasses__()) + tuple(ChannelImageAugmentation.__subclasses__())

all_factory_names = []
all_augmentation_names = []
for aug in leaf_augmentations:
    aug_name = aug.__name__
    distributions = aug.distributions
    aug_default_params = ", ".join((f"{k}={repr(v)}" for k, v in aug.distributions.items()))
    aug_params = ", ".join(f"{name}={name}" for name in aug.distributions.keys())
    factory_alias_def = f"def Random{aug_name}({aug_default_params}):\n\treturn {aug_name}.factory({aug_params})"
    print(factory_alias_def)
    exec(factory_alias_def)
    all_augmentation_names.append(aug_name)
    all_factory_names.append(f"def Random{aug_name}")


__all__ = [
    "Uniform",
    "Uniform2D",
    "Bernoulli",
    "Distribution",
    "Distribution2D",
    "Categorical",
    "Scale",
    "Rotate",
    "DeterministicImageAugmentation",
    "SpatialImageAugmentation",
    "ChannelImageAugmentation",
    "ImageAugmentationPipelineDataset",
    "WrapAugmentation",
    "AugmentationDataset",
    "ImageAugmentationPipelineDataset",
    "reset_all_seeds"
] + all_augmentation_names + all_factory_names
