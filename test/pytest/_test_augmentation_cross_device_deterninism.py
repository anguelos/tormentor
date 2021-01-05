# This test demonstrates non determinism across devices in pytorch.rand, affecting pytorch, and pytorch.distributions
import pytest
import torch
import tormentor


tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=100)
torch.manual_seed(0)

batch_size = 10
image_width = 224
image_height = 224


epsilon = .00000001


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon.
    """
    return ((t1 - t2) ** 2 > epsilon).view(-1).sum().item() == 0


in_img = torch.rand(3, image_width, image_height)
in_batch = in_img.unsqueeze(dim=0).repeat([batch_size, 1, 1, 1])


testable_augmentations = list(tormentor._leaf_augmentations)
testable_augmentations += [tormentor.AugmentationCascade.create([tormentor.Perspective, tormentor.Wrap])]
testable_augmentations += [tormentor.AugmentationChoice.create([tormentor.Perspective, tormentor.PlasmaBrightness])]


@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_cross_device_determinism(augmentation_cls):
    aug = augmentation_cls()

    # Assert cpu_gpu per sample
    cuda_img = in_img.to("cuda")
    assert all_similar(aug(in_img), aug(cuda_img).cpu())

    # Assert determinism per batch
    cuda_batch = in_batch.to("cuda")
    assert all_similar(aug(in_batch), aug(cuda_batch).cpu())

