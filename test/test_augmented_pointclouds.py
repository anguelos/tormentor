import torch
from matplotlib import pyplot as plt

import tormentor

tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=100)
torch.manual_seed(0)

augmentations_cls_list = []

for cls in [tormentor.SpatialImageAugmentation, tormentor.StaticImageAugmentation]:
    augmentations_cls_list += cls.__subclasses__()

epsilon = .00000001
plot_dbg = True
n_replicates = 1
n_pointclusters = 7
tolerance = 4  # pixels
width, height = 320, 240

def helper_create_pointcloud(width, height, n_points):
    pointcloud = torch.zeros([1 + 30, n_points]), torch.zeros([1 + 30, n_points])
    pointcloud[0][0, :] = torch.rand(n_points) * (width - 51) + 10
    pointcloud[1][0, :] = torch.rand(n_points) * (height - 51) + 10
    n = 1
    for dx in range(1, 41, 2):
        pointcloud[0][n, :] = pointcloud[0][0, :] + dx
        pointcloud[1][n, :] = pointcloud[1][0, :]
        n += 1
    for dy in range(1, 21, 2):
        pointcloud[0][n, :] = pointcloud[0][0, :]
        pointcloud[1][n, :] = pointcloud[1][0, :] + dy
        n += 1
    pointcloud = pointcloud[0].view(-1), pointcloud[1].view(-1)
    return pointcloud


def helper_render_pointcloud_to_channels(pc, width, height):
    img = torch.zeros(pc[0].size(0), height, width)
    for n in range(pc[0].size(0)):
        img[n, int(pc[1][n].round()), int(pc[0][n].round())] = 1
    return img

def helper_test_pointcoulds_as_images(augmentation_cls):
    pointcloud = helper_create_pointcloud(width, height, n_pointclusters)
    img = helper_render_pointcloud_to_channels(pointcloud, width, height)

    for replicate in range(n_replicates):
        augmentation = augmentation_cls()
        augmented_pointcloud, aug_img = augmentation(pointcloud, img)

        # Testing image augmentations agree when computed jointly with pointclouds and on their own they agree
        assert torch.all(torch.eq(aug_img, augmentation(img)))

        if plot_dbg:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.imshow(img.sum(dim=0))
            fig.suptitle(repr(augmentation))
            ax2.imshow(aug_img.sum(dim=0))

            ax3.plot(pointcloud[0].view(-1), pointcloud[1].view(-1), '.')
            ax4.plot(augmented_pointcloud[0].view(-1), augmented_pointcloud[1].view(-1), '.')

            [ax.set_xlim(ax1.get_xlim()) and ax.set_ylim(ax1.get_ylim()) for ax in [ax2, ax3, ax4]]
            plt.show()

        # sanity check for the testcase it self
        for n in range(pointcloud[0].size(0)):
            precise_x, precise_y = pointcloud[0][n], pointcloud[1][n]
            crude_x = torch.argmax(img[n, :, :].sum(dim=0)).item()
            crude_y = torch.argmax(img[n, :, :].sum(dim=1)).item()
            #if tolerance < precise_x < width - tolerance and tolerance < precise_y < height - tolerance:
            assert (crude_x - precise_x) ** 2 < tolerance ** 2
            assert (crude_y - precise_y) ** 2 < tolerance ** 2

        # Testing the pointcloud vs a onehot encoding of the points into channels
        for n in range(pointcloud[0].size(0)):
            precise_x, precise_y = augmented_pointcloud[0][n], augmented_pointcloud[1][n]
            crude_x = torch.argmax(aug_img[n, :, :].sum(dim=0)).item()
            crude_y = torch.argmax(aug_img[n, :, :].sum(dim=1)).item()
            if tolerance < precise_x < width - tolerance and tolerance < precise_y < height - tolerance:
                assert (crude_x - precise_x) ** 2 < tolerance ** 2
                assert (crude_y - precise_y) ** 2 < tolerance ** 2

def test_rotation():
    helper_test_pointcoulds_as_images(tormentor.Rotate)

def test_zoom():
    helper_test_pointcoulds_as_images(tormentor.Zoom)

def test_scale():
    helper_test_pointcoulds_as_images(tormentor.Scale)

def test_translate():
    helper_test_pointcoulds_as_images(tormentor.Translate)

def test_scale_translate():
    helper_test_pointcoulds_as_images(tormentor.ScaleTranslate)

def test_flip():
    helper_test_pointcoulds_as_images(tormentor.Flip)

def test_elastic_transform():
    helper_test_pointcoulds_as_images(tormentor.ElasticTransform)

def test_warp():
    helper_test_pointcoulds_as_images(tormentor.Wrap)
