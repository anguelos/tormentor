import pytest
import torch
import tormentor

def test_instanciation():
    assert isinstance(tormentor.RandomRotate(), tormentor.Rotate)

def test_representations():
    # Testing factory representations
    expected_repr = "RandomRotate.custom(radians=Uniform(value_range=(-3.14, 3.14)))"
    assert repr(tormentor.AugmentationFactory(tormentor.Rotate)) == expected_repr

    expected_repr = "(RandomBrightness.custom(brightness=Uniform(value_range=(-1.0, 1.0))) ^ "\
        "RandomRotate.custom(radians=Uniform(value_range=(-3.14, 3.14))))"
    assert repr(tormentor.RandomRotate ^ tormentor.RandomBrightness) == expected_repr

    expected_repr = "(RandomBrightness.custom(brightness=Uniform(value_range=(-1.0, 1.0))) | "\
        "RandomRotate.custom(radians=Uniform(value_range=(-3.14, 3.14))))"
    assert repr(tormentor.RandomRotate | tormentor.RandomBrightness) == expected_repr


def test_equality_between_factory_and_augmentation():
    # Testing equality and Factory / Augmentation
    assert (tormentor.RandomRotate | tormentor.Brightness) == (tormentor.RandomRotate | tormentor.RandomBrightness)



def test_resizing_factory():
    # testing resizing
    img = torch.rand(1, 3, 100, 100)
    crop_224 = tormentor.RandomPadCropTo
    rotate = tormentor.RandomRotate
    rotate_256_128 = tormentor.RandomRotate.new_size(width=256, height=128)
    crop_256 = tormentor.RandomRotate.new_size(width=256, height=256)
    assert crop_224()(img).size() == (1, 3, 224, 224)
    assert rotate()(img).size() == (1, 3, 100, 100)
    assert crop_256()(img).size() == (1, 3, 256, 256)
    assert rotate_256_128()(img).size() == (1, 3, 128, 256)

