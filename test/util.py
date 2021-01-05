import torch


def test_pointcoulds_as_images(augmentation, n_points, tolerance, size):
    pointcloud = torch.rand(1, 1, n_points) * (size - 1), torch.rand(1, 1, n_points) * (size - 1)
    img = torch.zeros(0, n_points, size, size)
    for n in n_points:
        img[0, n, int(pointcloud[0][0, 0, n].round()), int(pointcloud[0][0, 0, n].round())] = 1
    aug_img = augmentation(img)
    aug_pc =  augmentation.forward_pointcloud(pointcloud)
    for n in range(n_points):
        crude_x = torch.argmax(aug_img[0, n, :, :].sum(dim=1))
        crude_y = torch.argmax(aug_img[0, n, :, :].sum(dim=0))
        assert (crude_x-aug_pc[0][n]) ** 2 < tolerance ** 2 and (crude_y-aug_pc[1][n]) ** 2 < tolerance ** 2
