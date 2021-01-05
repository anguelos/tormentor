import pytest
import torch
import tormentor

assert isinstance(tormentor.RandomRotate(), tormentor.Rotate)

# Testing factory representations
expected_repr = "RandomRotate.custom(radians=Uniform(value_range=(-3.14, 3.14)))"
assert repr(tormentor.AugmentationFactory(tormentor.Rotate)) == expected_repr

expected_repr = "(RandomBrightness.custom(brightness=Uniform(value_range=(-1.0, 1.0))) ^ "\
    "RandomRotate.custom(radians=Uniform(value_range=(-3.14, 3.14))))"
assert repr(tormentor.RandomRotate ^ tormentor.RandomBrightness) == expected_repr

expected_repr = "(RandomBrightness.custom(brightness=Uniform(value_range=(-1.0, 1.0))) | "\
    "RandomRotate.custom(radians=Uniform(value_range=(-3.14, 3.14))))"
assert repr(tormentor.RandomRotate | tormentor.RandomBrightness) == expected_repr
