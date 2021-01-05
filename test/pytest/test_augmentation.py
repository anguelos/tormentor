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
def test_determinism(augmentation_cls):
    # Assert determinism per sample
    aug = augmentation_cls()
    assert all_similar(aug(in_img), aug(in_img))

    # Assert determinism per batch
    aug = augmentation_cls()
    assert all_similar(aug(in_batch), aug(in_batch))

    # Assert determinism per sample
    aug = augmentation_cls()
    assert all_similar(aug(in_img.to("cuda")), aug(in_img.to("cuda")))

    # Assert determinism per batch
    aug = augmentation_cls()
    assert all_similar(aug(in_batch.to("cuda")), aug(in_batch.to("cuda")))


# pytorch.rand is not deterministic across devices, when pytorch fixes this, this testcase will be enabled.
## This fails for plasma and composite augmentations
#testable_augmentations = list(tormentor._leaf_augmentations)
#testable_augmentations += [tormentor.AugmentationCascade.create([tormentor.Perspective, tormentor.Wrap])]
#testable_augmentations += [tormentor.AugmentationChoice.create([tormentor.Perspective, tormentor.PlasmaBrightness])]
#@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
#def test_cross_device_determinism(augmentation_cls):
#    aug = augmentation_cls()

#    # Assert cpu_gpu per sample
#    cuda_img = in_img.to("cuda")
#    assert all_similar(aug(in_img), aug(cuda_img).cpu())

#    # Assert determinism per batch
#    cuda_batch = in_batch.to("cuda")
#    assert all_similar(aug(in_batch), aug(cuda_batch).cpu())


# any augmentation with a high probability of being an identity function should be removed
testable_augmentations = list(tormentor._leaf_augmentations - {tormentor.Flip, tormentor.Invert, tormentor.PadTo, tormentor.PadCropTo, tormentor.CropTo})
testable_augmentations += [tormentor.AugmentationCascade.create([tormentor.Perspective, tormentor.Wrap])]
testable_augmentations += [tormentor.AugmentationChoice.create([tormentor.Perspective, tormentor.PlasmaBrightness])]


@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_non_identity(augmentation_cls):
    # Are two different augmentations really different?
    aug1 = augmentation_cls()
    aug2 = augmentation_cls()
    assert not all_similar(aug1(in_img), aug2(in_img))

    # is one augmentation doing different things to other samples in a batch
    out_img = aug1(in_img)
    out_batch = aug1(in_batch)
    for n in range(1, batch_size):
        assert not all_similar(out_img, out_batch[n, :, :, :])
