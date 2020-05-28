import types
import numbers

import torch

from .base_augmentation import DeterministicImageAugmentation, SpatialImageAugmentation, ChannelImageAugmentation


def _get_augmentation_subjects_from_dataset(dataset, where, augmentation):
    """Guesses which images are image tensors.
    """
    sample = dataset[0]
    if where is "guess":
        if isinstance(augmentation, SpatialImageAugmentation):
            width, height = sample[0].size()[-2:]
            apply_on = []
            for t in sample:
                if isinstance(t, torch.Tensor) and len(t.size()) >= 2 and t.size(-2) == width and t.size(-1) == height:
                    apply_on.append(True)
                else:
                    apply_on.append(False)
            return tuple(apply_on)
        elif isinstance(augmentation, ChannelImageAugmentation):
            # channel augmentations are only applicable on the input image
            return (True,) + (False,) * (len(dataset[0]) - 1)

    elif where == "all":
        return (True,) * len(dataset[0])
    elif where == "first" or where == 0:
        return (True,) + (False,) * (len(dataset[0]) - 1)
    elif where == "last" or where == -1:
        return (True,) + (False,) * (len(dataset[0]) - 1)
    elif isinstance(where, numbers.Integral):
        res = [False] * len(dataset[0])
        res[where] = True
        return tuple(res)
    elif isinstance(where, (list, tuple)):
        return tuple(where)
    else:
        raise ValueError()


class AugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentation_factory, apply_on="guess",append_input_map=False):
        self.dataset = dataset
        self.augmentation_factory = augmentation_factory
        self.apply_on = _get_augmentation_subjects_from_dataset(dataset, where=apply_on, augmentation=augmentation_factory())
        self.append_input_map = append_input_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        augmentation = self.augmentation_factory.create_persistent()
        sample = self.dataset[item]
        sample = tuple([augmentation(sample[n]) if self.apply_on[n] else sample[n] for n in range(len(sample))])
        if self.append_input_map:
            sample = sample + (augmentation(torch.ones_like(sample[0])),)
        return sample


class ImageAugmentationPipelineDataset(torch.utils.data.Dataset):
    """Augments a pytorch dataset with some augmentations

        A dataset is assumed to be an oredered collection of 4D tensors (images), tensors of other dimensions and
        numbers float or integers. Images are assumed to be (Batch x Channels x Width x Height) and to always have the
        batch dimension with a size of 1. For segmentation data to be properly augmented, pixel labels must be encoded
        as one-hot encoding along the channel dimension.
    """

    def __init__(self, dataset, augmentations, apply_on="guess", occurrence_prob=1.0, append_input_map=False,
                 train=None):
        """Augments a pytorch dataset with some augmentations.

        :param dataset: A pytorch dataset to be wrapped and augmented.
        :param augmentations: One or more augmentation class or a factory generating instances of deterministic
            augmetions to be applied on each sample.
        :param apply_on: One of  ["all", "first", "last", "guess"] or a tuple or list of booleans. It describes on
            witch elements of a sample to apply the augmentation on. Guessing is done on the the assumption that the
            first item in a sample is an image (4D tensor) and the last two dimensions are width and height. All tensors
            having the last two dimensions with this size are assumed to be tensors to be augmented.
        :param occurrence_prob: The probability of augmenting the sample.
        :param append_input_map: If True, the sample will be appended by an image of ones wich will then be passed in
            the augmentation; after augmentation these images endout what is the contribution of the original image to
            each pixel after the augmentation.
        """
        if train is None:
            try:
                self.train = dataset.train
            except AttributeError:
                self.train = None
        else:
            self.train = train
        self.apply_on = _get_augmentation_subjects_from_dataset(dataset, where=apply_on, augmentation=augmentations[0]())
        self.append_input_map = append_input_map
        if append_input_map:
            apply_on.append(True)
        self.apply_on = apply_on
        self.dataset = dataset
        self.occurrence_prob = occurrence_prob
        if not isinstance(augmentations, (list, tuple)):
            augmentations = (augmentations,)
        self.augmentation_factories = []

        for augmentation in augmentations:
                if isinstance(augmentation, types.LambdaType):
                    self.augmentation_factories.append(augmentation)
                elif issubclass(augmentation, SpatialImageAugmentation):
                    self.augmentation_factories.append(augmentation.factory())
                else:
                    raise ValueError(
                        "augmentation must be either a DeterministicImageAugmentation or a lambda producing them.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = list(self.dataset[item])
        if self.append_input_map:
            sample.append(torch.ones_like(sample[0]))
        if torch.rand(1).item() > self.occurrence_prob:
            return sample
        else:
            augmentations = [f() for f in self.augmentation_factories]
            augmented_sample = []
            for should_apply, sample_tensor in zip(self.apply_on, sample):
                if should_apply:
                    for augmentation in augmentations:
                        sample_tensor = augmentation(sample_tensor)
                    augmented_sample.append(sample_tensor)
                else:
                    augmented_sample.append(sample_tensor)
            return augmented_sample
