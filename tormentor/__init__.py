r"""Segmentation aware image augmentations

This module realises basic utilities for image augmentations.

General design principles:
    - Images are always 3D tensors [CxHxW], batches are always 4D tensors [BxCxHxW]
    - Augmentations are always a applied on a single sample
    - The first dimension of the 4D tensor is always 1

"""

from .random import Uniform, Bernoulli, Distribution, Categorical
from .base_augmentation import StaticImageAugmentation, SpatialImageAugmentation, DeterministicImageAugmentation, AugmentationChoice, AugmentationCascade
from .spatial_augmentations import *
from .color_augmentations import *
from .augmented_dataset import AugmentedDs, AugmentedCocoDs
from .wrap import Wrap, ShredAugmentation
from .backgrounds import ConstantBackground, NormalNoiseBackground, UniformNoiseBackground, PlasmaBackground
from .util import debug_pattern

reset_all_seeds = DeterministicImageAugmentation.reset_all_seeds

leaf_augmentations = tuple(SpatialImageAugmentation.__subclasses__()) + tuple(StaticImageAugmentation.__subclasses__())

all_factory_names = []
all_augmentation_names = []
for aug in leaf_augmentations:
    aug_name = aug.__name__
    distributions = aug.distributions
    aug_default_params = ", ".join((f"{k}={repr(v)}" for k, v in aug.distributions.items()))
    aug_params = ", ".join(f"{name}={name}" for name in aug.distributions.keys())
    factory_alias_def = f"def Random{aug_name}({aug_default_params}):\n\treturn {aug_name}.factory({aug_params})"
    exec(factory_alias_def)
    all_augmentation_names.append(aug_name)
    all_factory_names.append(f"def Random{aug_name}")

choice = AugmentationChoice.create
cascade = AugmentationCascade.create

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
    "ChannelImageAugmentation",
    "ImageAugmentationPipelineDataset",
    "WrapAugmentation",
    "AugmentedDs",
    "AugmentedCocoDs",
    "ImageAugmentationPipelineDataset",
    "reset_all_seeds",
    "ConstantBackground",
    "UniformNoiseBackground",
    "NormalNoiseBackground",
    "PlasmaBackground",
    "ShredAugmentation",
    "choice",
    "cascade"
] + all_augmentation_names + all_factory_names
