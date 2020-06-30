import torch

import tormentor

from matplotlib import pyplot as plt

tormentor.DeterministicImageAugmentation.reset_all_seeds(global_seed=100)
torch.manual_seed(7)

base_cls = tormentor.DeterministicImageAugmentation

intermediary_cls = base_cls.__subclasses__()
augmentations_cls_list = []


for cls in [tormentor.SpatialImageAugmentation, tormentor.ChannelImageAugmentation]:
    augmentations_cls_list += cls.__subclasses__()


#augmentations_cls_list = [tormentor.Rotate.override_distributions(radians=tormentor.Uniform((0,3.14/2)))]
augmentations_cls_list = [tormentor.Zoom.override_distributions(scales=tormentor.Uniform((.75,.76)))]

epsilon = .00000001


def helper_pointcoulds_as_images(augmentation, n_points, tolerance, size):
    pointcloud = torch.zeros([1, 1+10, n_points]), torch.zeros([1, 1+10, n_points])
    pointcloud[0][0,0,:]=torch.rand(n_points)*size[0]-4
    pointcloud[1][0,0,:]=torch.rand(n_points)*size[1]-4
    n = 1
    for dx in range(1, 8):
        pointcloud[0][0, n, :] = pointcloud[0][0, 0, :] + dx
        pointcloud[1][0, n, :] = pointcloud[1][0, 0, :]
        n+=1
    for dy in range(1, 4):
        pointcloud[0][0, n, :] = pointcloud[0][0, 0, :]
        pointcloud[1][0, n, :] = pointcloud[1][0, 0, :] + dy
        n += 1
    pointcloud = pointcloud[0].view([1,1,-1]), pointcloud[1].view([1,1,-1])
    n_points = pointcloud[0].size(2)

    img = torch.zeros(1, n_points, size[0], size[1])

    for n in range(n_points):
        img[0, n, int(pointcloud[0][0, 0, n].round()), int(pointcloud[1][0, 0, n].round())] = 1

    aug_img = augmentation(img)
    aug_pc = augmentation(pointcloud, img)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.imshow(img[0, :, :, :].sum(dim=0))
    ax2.imshow(aug_img[0, :, :, :].sum(dim=0))
    ax3.plot(pointcloud[0].view(-1), pointcloud[1].view(-1),'.')
    ax4.plot(aug_pc[0].view(-1), aug_pc[1].view(-1),'.')
    [ax.set_xlim(ax1.get_xlim()) and ax.set_ylim(ax1.get_ylim()) for ax in [ax2,ax3,ax4]]
    plt.show()


    # sanity check for the testcase
    for n in range(n_points):
        crude_x = torch.argmax(img[0, n, :, :].sum(dim=1)).item()
        crude_y = torch.argmax(img[0, n, :, :].sum(dim=0)).item()
        assert (crude_x-pointcloud[0][0,0,n]) ** 2 < tolerance ** 2
        assert (crude_y-pointcloud[1][0,0,n]) ** 2 < tolerance ** 2




    # Testing the pointcloud vs a onehot encoding of the points into channels
    for n in range(n_points):
        crude_x = torch.argmax(aug_img[0, n, :, :].sum(dim=1)).item()
        crude_y = torch.argmax(aug_img[0, n, :, :].sum(dim=0)).item()
        assert (crude_y-aug_pc[1][0,0,n]) ** 2 < tolerance ** 2
        assert (crude_x-aug_pc[0][0,0,n]) ** 2 < tolerance ** 2



def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon.
    """
    return ((t1 - t2) ** 2 > epsilon).view(-1).sum() == 0

def test_point_clouds():
    for augmentation_cls in augmentations_cls_list:
        for replicate in range(1):
            augmentation = augmentation_cls()
            print(augmentation)
            helper_pointcoulds_as_images(augmentation, 3, 1., [200, 100])


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
