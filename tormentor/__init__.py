r"""Segmentation aware image augmentations

This module realises basic utilities for image augmentations.

Augmentations are either spatial or

General design principles:
    - Images are always 4D tensors
    - Augmentations are always a applied on a single sample
    - The first dimension of the 4D tensor is always 1

"""
from .base_augmentation import ChannelImageAugmentation, SpatialImageAugmentation, DeterministicImageAugmentation
from .spatial_augmentations import *
from .augmented_dataset import ImageAugmentationPipelineDataset
from .wrap import WrapAugmentation
from .augmented_dataset import AugmentationDataset, ImageAugmentationPipelineDataset
reset_all_seeds = DeterministicImageAugmentation.reset_all_seeds

__all__ = [
    "DeterministicImageAugmentation",
    "SpatialImageAugmentation",
    "ChannelImageAugmentation",
    "ScaleAndPadAsNeeded",
    "CropPadAsNeeded",
    "EraseRectangle",
    "Flip",
    "Scale",
    "Rotate",
    "ImageAugmentationPipelineDataset",
    "WrapAugmentation",
    "AugmentationDataset",
    "ImageAugmentationPipelineDataset",
    "reset_all_seeds"
]
