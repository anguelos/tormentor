import pytest
import torch
import kornia.tormentor as tormentor


def test_scale_and_pad_as_needed():
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 10:20, 5:20] = 1
    default_factory = tormentor.ScaleAndPadAsNeeded.factory()
    aug_times4 = default_factory()
    assert aug_times4(image).size() == torch.Size([1, 3, 128, 128])
    assert .95 < aug_times4(image).sum() / (image.sum() * 4 ** 2) < 1.05
    aug_half_v1 = tormentor.ScaleAndPadAsNeeded.factory(16,16)()
    aug_half_v2 = tormentor.ScaleAndPadAsNeeded.factory(16,16)()
    assert .95 < aug_half_v1(image).sum() / (150 * 3 * .5 ** 2) < 1.05
    # scaling down makes all augmentations of the same scale
    assert (aug_half_v1(image) - aug_half_v2(image)).abs().sum() < .000001
