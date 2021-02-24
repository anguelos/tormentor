import pytest
import torch
import tormentor
import PIL
import time
from matplotlib import pyplot as plt

tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=0)
torch.manual_seed(0)

epsilon = .1
tolarated_wrong_pixels = .1

batch_size = 10
image_width = 224
image_height = 224
n_channels = 10

def most_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon for most pixels.
    """
    t1, t2 = t1.float(), t2.float()
    print("error:",((t1 - t2) ** 2 ).mean())
    print("tolerance:",((t1 - t2) ** 2 > epsilon).view(-1).float().mean().item())
    return ((t1 - t2) ** 2 > epsilon).view(-1).float().mean().item() < tolarated_wrong_pixels


testable_augmentations = list(tormentor._leaf_augmentations)
testable_augmentations += [tormentor.AugmentationCascade.create([tormentor.Perspective, tormentor.Wrap])]
testable_augmentations += [tormentor.AugmentationChoice.create([tormentor.Perspective, tormentor.PlasmaBrightness])]

batch_labels = torch.zeros([batch_size, image_height, image_width], dtype=torch.int64)
batch_onehots = torch.zeros([batch_size, n_channels, image_height, image_width], dtype=torch.float)
for n in range(1, n_channels):
    channel = n
    batch_onehots[:, channel, n * 8: -n * 8, n * 8: -n * 8] = 1
    batch_onehots[:, channel - 1, n * 8: -n * 8, n * 8: -n * 8] = 0
    batch_labels[:, n * 8 : -n*8, n * 8 : -n*8 ] = n
batch_onehots[:, 0, :, :] = (batch_onehots[:,1:, :,:].sum(dim=1)<=0).float()




@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_batch(augmentation_cls):
    aug = augmentation_cls()
    augmented_labels = aug(batch_labels, is_mask=True)
    augmented_onehots = aug(batch_onehots, is_mask=True)
    for channel in range(n_channels):
        #f,ax = plt.subplots(2,1)
        #ax[0].imshow(augmented_labels[0, :,:]==channel)
        #ax[1].imshow(augmented_onehots[0, channel, :, :])
        #plt.show()
        assert most_similar(augmented_onehots[:, channel ,:, :], augmented_labels == channel)

@pytest.mark.parametrize("augmentation_cls", [cls for cls in testable_augmentations])
def test_sample(augmentation_cls):
    aug = augmentation_cls()
    labels = batch_labels[0, :, :]
    onehots = batch_onehots[0, :, : , :]
    augmented_labels = aug(labels, is_mask=True)
    augmented_onehots = aug(onehots, is_mask=True)
    for channel in range(n_channels):
        assert most_similar(augmented_onehots[channel ,:, :], augmented_labels == channel)
