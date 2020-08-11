import pytest
import torch
import tormentor
from matplotlib import pyplot as plt


tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=100)
torch.manual_seed(0)

batch_size = 10
image_width = 128
image_height = 128

plot_dbg = False

# The epsilon is high because we compare sums over images.
epsilon = .001


def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon.
    """
    return ((t1 - t2) ** 2 > epsilon).view(-1).sum().item() == 0


in_img_sf = tormentor.create_sampling_field(image_width, image_height, batch_size=0)
in_img = torch.rand(3, image_width, image_height)
in_batch = in_img.unsqueeze(dim=0).repeat([batch_size, 1, 1, 1])

@pytest.mark.parametrize("pad_size", [(100, 400), (200, 100), (100, 100), (200,200), (128, 128)])
def test_padto(pad_size):
    width, height = pad_size
    aug = tormentor.PadTo.new_size(*pad_size)()
    new_img = aug(in_img)
    assert new_img.size(1) >= in_img.size(1) and new_img.size(2) >= in_img.size(2)
    assert new_img.size(1) >= height and new_img.size(2) >= width
    assert all_similar(new_img.sum(), in_img.sum())

    new_img_sf = aug(in_img_sf)
    new_sf_img = tormentor.apply_sampling_field(in_img, new_img_sf)
    if plot_dbg:
        f, ax = plt.subplots(2,3)
        ax[0][0].imshow(new_sf_img[0, :, :])
        print("new_sf_img" ,new_sf_img[0, :, :].mean())
        ax[1][0].imshow(new_img[0, :, :])

        ax[0][1].imshow(in_img_sf[0], vmin=-1., vmax=1.);ax[1][1].imshow(in_img_sf[1], vmin=-1., vmax=1.)
        print("in_img_sf", in_img_sf[0].mean())

        ax[0][2].imshow(new_img_sf[0], vmin=-1., vmax=1.);ax[1][2].imshow(new_img_sf[1], vmin=-1., vmax=1.)
        print("new_img_sf", new_img_sf[0].mean())

        plt.show()
    assert all_similar(new_sf_img, new_img)


@pytest.mark.parametrize("crop_size", [(100, 100), (200, 100), (200, 100), (200, 200), (128, 128)])
def test_crop(crop_size):
    width, height = crop_size
    aug = tormentor.CropTo.new_size(*crop_size)()
    new_img = aug(in_img)
    assert new_img.size(1) <= in_img.size(1) and new_img.size(2) <= in_img.size(2)
    assert new_img.size(1) <= height and new_img.size(2) <= width

    new_img_sf = aug(in_img_sf)
    new_sf_img = tormentor.apply_sampling_field(in_img, new_img_sf)
    if plot_dbg:
        f, ax = plt.subplots(2,3)
        ax[0][0].imshow(new_sf_img[0,:,:])
        print("new_sf_img" ,new_sf_img[0,:,:].mean())
        ax[1][0].imshow(new_img[0,:,:])

        ax[0][1].imshow(in_img_sf[0], vmin=-1., vmax=1.);ax[1][1].imshow(in_img_sf[1], vmin=-1., vmax=1.)
        print("in_img_sf", in_img_sf[0].mean())

        ax[0][2].imshow(new_img_sf[0], vmin=-1., vmax=1.);ax[1][2].imshow(new_img_sf[1], vmin=-1., vmax=1.)
        print("new_img_sf", new_img_sf[0].mean())

        plt.show()
    assert all_similar(new_sf_img, new_img)
