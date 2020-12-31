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


testable_augmentations = list(tormentor._leaf_augmentations)
testable_augmentations += [tormentor.AugmentationCascade.create([tormentor.Perspective, tormentor.Wrap])]
testable_augmentations += [tormentor.AugmentationChoice.create([tormentor.Perspective, tormentor.PlasmaBrightness])]



def augment_batch(batch, aug, device):
    return [aug(batch[0]).to(device)] + batch[1:]


def augment_sample(sample, aug, device):
    return (aug(sample[0]).to(device),) + sample[1:]


aug_ds = tormentor.AugmentedDs(ds, tormentor.RandomRotate, computation_device="cpu",
                               augment_sample_function=augment_sample)


batch_aug_dl = tormentor.AugmentedDataLoader(dl, tormentor.RandomRotate, computation_device="cpu",
                               augment_batch_function=augment_batch)
sample_aug_dl = torch.utils.data.DataLoader(aug_ds, batch_size=1)


@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_determinism(augmentation_cls):
    aug_ds = tormentor.AugmentedDs(ds, tormentor.AugmentationFactory(augmentation_cls), computation_device="cpu",
                                   augment_sample_function=augment_sample)
    batch_aug_dl = tormentor.AugmentedDataLoader(dl, tormentor.AugmentationFactory(augmentation_cls), computation_device="cpu",
                                                 augment_batch_function=augment_batch)
    sample_aug_dl = torch.utils.data.DataLoader(aug_ds, batch_size=1)

    torch.manual_seed(0)
    tormentor.reset_all_seeds()
    per_batch = torch.cat([batch[0] for batch in batch_aug_dl])

    torch.manual_seed(0)
    tormentor.reset_all_seeds()
    per_sample = torch.cat([batch[0] for batch in sample_aug_dl])
    assert all_similar(per_batch, per_sample)

