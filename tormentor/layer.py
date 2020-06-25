import torch
from .base_augmentation import DeterministicImageAugmentation


class AugmentationLayer(torch.nn.Module):
    """Generates deterministic augmentations.

    It is a serializable generator that can override default random get_distribution_parameters.
    """
    def __init__(self, image, **kwargs):
        super().__init__()
        assert set(kwargs.keys()) <= set(augmentation_cls.distributions.keys())
        self.augmentation_distributions = {k: v.copy() for k, v in augmentation_cls.distributions.items()}
        overridden_distributions = {k: v for k, v in kwargs.items() if k in augmentation_cls.distributions.keys()}
        self.augmentation_distributions.update(overridden_distributions)
        self.augmentation_distributions = torch.nn.ModuleDict(self.augmentation_distributions)
        self.augmentation_cls = augmentation_cls

    def forward(self, x):
        if self.training:
            return self.create_persistent()(x)
        else:
            return x

    def create_persistent(self):
        return self.augmentation_cls()
