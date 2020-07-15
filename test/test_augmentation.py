import torch

import tormentor


tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=100)
torch.manual_seed(7)

base_cls = tormentor.DeterministicImageAugmentation

intermediary_cls = base_cls.__subclasses__()
augmentations_cls_list = []


for cls in [tormentor.SpatialImageAugmentation, tormentor.StaticImageAugmentation]:
    augmentations_cls_list += cls.__subclasses__()


epsilon = .00000001


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon.
    """
    return ((t1 - t2) ** 2 > epsilon).view(-1).sum() == 0


def test_minimum_requirement():
    for augmentation_cls in augmentations_cls_list:

        # Every augmentation must define at least one of forward_batch_img, forward_sample_img
        assert base_cls.forward_batch_img is not augmentation_cls.forward_batch_img or base_cls.forward_sample_img is not augmentation_cls.forward_sample_img

        # Assert determinism per sample
        aug = augmentation_cls()
        img = torch.rand(3, 224, 224)
        assert all_similar(aug(img), aug(img))

        # Assert determinism per batch
        aug = augmentation_cls()
        img = torch.rand(10, 3, 224, 224)
        assert all_similar(aug(img), aug(img))

        # Augmentation states must be available after augmentation has been run once.
        #aug = augmentation_cls()
        #img1 = torch.rand(3, 224, 224)
        #aug(img1)
        #for state_name in augmentation_cls._state_names:
        #    assert getattr(aug, state_name) is not None


def test_hard_requirement():
    # these tests should be perceived as warnings and don't make sense for all augmentations
    for augmentation_cls in augmentations_cls_list:
        # Augmentation is defining both of forward_batch_img and forward_sample_img
        assert base_cls.forward_batch_img is not augmentation_cls.forward_batch_img
        assert base_cls.forward_sample_img is not augmentation_cls.forward_sample_img

        # Was the aug_distributions decorator used?
        assert len(augmentation_cls._state_names) > 0

        # Are two different augmentations really different?
        img = torch.rand(3, 224, 224)
        aug1 = augmentation_cls()
        aug2 = augmentation_cls()
        assert not all_similar(aug1(img), aug2(img))
