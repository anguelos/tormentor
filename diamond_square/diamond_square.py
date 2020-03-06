import math

import torch


def diamond_square(desired_size=None, output_range=None, N=None, roughnes=.5, device=None, seed_img=None):
    """Renders a plasma fractal with the diamond square algorithm

    :param N:
    :param roughnes:
    :param device:
    :param seed_img:
    :return:
    """
    # TODO (anguelos) apply these parameters on the results
    if desired_size is not None:
        assert N is None
        N = max(math.log2(desired_size[0]), math.log2(desired_size[1])) + 1
    else:
        assert N is not None
    if output_range is None:
        output_range = [0.0, 1.0]

    with torch.no_grad():
        if seed_img is None and device is None:
            device = "cpu"
        elif seed_img is not None and device is None:
            device = seed_img.device

        if seed_img is None:
            seed_img = torch.rand(1, 1, 3, 3).to(device)

            # first diamond
            rnd_range = 1.0 * roughnes
            seed_img[:, :, 1, 1] = .25 * (
                        seed_img[:, :, 0, 0] + seed_img[:, :, 2, 0] + seed_img[:, :, 0, 2] + seed_img[:, :, 2, 2]) + \
                                   torch.rand(1) * roughnes - roughnes / 2
            # first square
            seed_img[:, :, 0, 1] = .3333 * (
                        seed_img[:, :, 0, 0] + seed_img[:, :, 0, 2] + seed_img[:, :, 1, 1]) + torch.rand(
                1) * roughnes - roughnes / 2
            seed_img[:, :, 1, 0] = .3333 * (
                        seed_img[:, :, 0, 0] + seed_img[:, :, 2, 0] + seed_img[:, :, 1, 1]) + torch.rand(
                1) * roughnes - roughnes / 2
            seed_img[:, :, 2, 1] = .3333 * (
                        seed_img[:, :, 2, 0] + seed_img[:, :, 2, 2] + seed_img[:, :, 1, 1]) + torch.rand(
                1) * roughnes - roughnes / 2
            seed_img[:, :, 1, 2] = .3333 * (
                        seed_img[:, :, 0, 2] + seed_img[:, :, 2, 2] + seed_img[:, :, 1, 1]) + torch.rand(
                1) * roughnes - roughnes / 2
            rnd_range = rnd_range * roughnes
            N -= 1
        else:
            while len(seed_img.size()) < 4:
                seed_img = seed_img.unsqueeze(dim=0)
            assert len(seed_img.size()) == 4
            assert seed_img.size(2) == seed_img.size(3) and 2 ** math.log2((seed_img.size(3) - 1)) + 1 == seed_img.size(
                3)
            seed_img = seed_img.to(device)
            rnd_range = 1.0 * roughnes
        img = seed_img
        for n in range(1, N + 1):
            img = one_diamond_one_square(img, rnd_range)
            rnd_range *= roughnes
        return img#[0, 0, :, :]


def one_diamond_one_square(img, roughnes):
    """Doubles the image resolution by applying a single diamond square steps

    Recursive application of this method creates plasma fractals.
    Attention! The function is differentiable and gradients are computed as well.
    If this function is run in the usual sense, it is more efficient if it is run in a no_grad()

    :param img: A 4D tensor where dimensions are Batch, Channel, Width, Height. Width and Height must both be 2^N+1 and
        Batch and Channels should in the usual case be 1.
    :param roughness: A float  number in [0,1] controlling the randomness created pixels get. I the usual case, it is
        halved at every applycation of this function.
    :return: A tensor on the same device as img with the same channels as img and width, height of 2^(N+1)+1
    """
    # TODO (anguelos) test multi channel and batch size > 1

    diamond_kernel = [[.25, 0., .25], [0., 0., 0.], [.25, 0., .25]]
    square_kernel = [[0., .25, 0.], [.25, 0., .25], [0., .25, 0.]]
    square_kernel = torch.tensor(square_kernel).unsqueeze(dim=0).unsqueeze(dim=0).to(img.device)
    diamond_kernel = torch.tensor(diamond_kernel).unsqueeze(dim=0).unsqueeze(dim=0).to(img.device)
    step = 2
    new_img = torch.zeros([1, 1, 2 * (img.shape[2] - 1) + 1, 2 * (img.shape[3] - 1) + 1], device=img.device)
    new_img[:1, :1, ::step, ::step] = img

    pad_compencate = torch.ones_like(new_img)
    pad_compencate[:, :, :, 0] = 1 / .75
    pad_compencate[:, :, :, -1] = 1 / .75
    pad_compencate[:, :, 0, :] = 1 / .75
    pad_compencate[:, :, -1, :] = 1 / .75

    rnd_img = torch.rand_like(new_img) * roughnes

    # diamond
    diamond_regions = torch.nn.functional.conv2d(new_img, diamond_kernel, padding=1)
    diamond_centers = (diamond_regions > 0).float()
    # TODO (anguelos) make sure diamond_regions*diamond_centers is needed
    new_img = new_img + (1 - roughnes) * diamond_regions * diamond_centers + diamond_centers * rnd_img

    # square
    square_regions = torch.nn.functional.conv2d(new_img, square_kernel, padding=1) * pad_compencate
    square_centers = (square_regions > 0).float()
    # TODO (anguelos) make sure square_centers*square_regions is needed
    new_img = new_img + square_centers * rnd_img + (1 - roughnes) * square_centers * square_regions

    return new_img
