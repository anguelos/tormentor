import pytest
import torch
import tormentor


tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=100)
torch.manual_seed(0)

dataset_size = 10
image_width = 224
image_height = 224


epsilon = .00000001


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon.
    """
    return ((t1 - t2) ** 2 > epsilon).view(-1).sum().item() == 0


ds = [(torch.rand(3,image_height, image_width), 0) for _ in range(dataset_size)]
dl = torch.utils.data.DataLoader(ds, batch_size=1)


testable_augmentations = list(tormentor._leaf_augmentations - {tormentor.Flip, tormentor.Invert, tormentor.PadTo, tormentor.PadCropTo, tormentor.CropTo})
testable_augmentations += [tormentor.AugmentationCascade.create([tormentor.Perspective, tormentor.Wrap])]
testable_augmentations += [tormentor.AugmentationChoice.create([tormentor.Perspective, tormentor.PlasmaBrightness])]


aug_ds = tormentor.AugmentedDs(ds, tormentor.RandomRotate, computation_device="cpu")
batch_aug_dl = tormentor.AugmentedDataLoader(dl, tormentor.RandomRotate, computation_device="cpu")
sample_aug_dl = torch.utils.data.DataLoader(aug_ds, batch_size=1)


@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_sample_batch(augmentation_cls):
    aug_ds = tormentor.AugmentedDs(ds, tormentor.AugmentationFactory(augmentation_cls), computation_device="cpu")
    batch_aug_dl = tormentor.AugmentedDataLoader(dl, tormentor.AugmentationFactory(augmentation_cls), computation_device="cpu")
    sample_aug_dl = torch.utils.data.DataLoader(aug_ds, batch_size=1)

    torch.manual_seed(0)
    tormentor.reset_all_seeds()
    per_batch = torch.cat([batch[0] for batch in batch_aug_dl])

    torch.manual_seed(0)
    tormentor.reset_all_seeds()
    per_sample = torch.cat([batch[0] for batch in sample_aug_dl])
    assert all_similar(per_batch, per_sample)

    #testing __len__ for dataset and dataloader
    assert len(batch_aug_dl) == len(sample_aug_dl) == len(aug_ds)

@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_batch_size(augmentation_cls):
    dl = torch.utils.data.DataLoader(ds, batch_size=10)
    batch_aug_dl = tormentor.AugmentedDataLoader(dl, augmentation_cls, computation_device="cpu")
    input, _ = next(iter(dl))
    aug_input, _ = next(iter(batch_aug_dl))
    assert input.size() == aug_input.size()
    assert input.size() == aug_input.size()

