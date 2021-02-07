import math
import torch

# For avoiding division by zero
_epsilon = .00000001


# noinspection PyPackageRequirements
def diamond_square(width_height=None, replicates=1, output_range=None, recursion_depth=None, roughness=.5,
                   device=None, seed_img=None, return_2d=False, output_deviation_mean=None, random=torch.rand) -> torch.Tensor:
    """Generates Plasma Fractal Images

    Args:
        width_height: A tuple of integers with the width and height of the image to be generated.
        replicates: An integer indicating how may diamond squares to compute in total.
        output_range: A tuple of floats indicating the range to which the fractals will be normalised. Must be None if
            output_deviation_mean is not None. If None
        recursion_depth:
        roughness:
        device:
        seed_img:
        return_2d:
        output_deviation_mean:

    Returns:

    """

    if width_height is not None:
        assert recursion_depth is None
        recursion_depth = max(math.log2(width_height[0] - 1), math.log2(width_height[1] - 1))
        recursion_depth = int(math.ceil(recursion_depth))
    else:
        assert recursion_depth is not None
    if output_range is None and output_deviation_mean is None:
        output_range = [0.0, 1.0]

    with torch.no_grad():
        if seed_img is None and device is None:
            device = "cpu"
        elif seed_img is not None and device is None:
            device = seed_img.device

        if seed_img is None:
            # TODO (anguelos) create non 3x3 seed image
            seed_img = torch.rand(replicates, 1, 3, 3).to(device)

            # first diamond
            rnd_range = 1.0 * roughness
            seed_img[:, :, 1, 1] = .25 * (
                    seed_img[:, :, 0, 0] + seed_img[:, :, 2, 0] + seed_img[:, :, 0, 2] + seed_img[:, :, 2, 2]) + \
                                   torch.rand(1, device=device) * roughness - roughness / 2
            # first square
            seed_img[:, :, 0, 1] = .3333 * (
                    seed_img[:, :, 0, 0] + seed_img[:, :, 0, 2] + seed_img[:, :, 1, 1]) + torch.rand(
                1, device=device) * roughness - roughness / 2
            seed_img[:, :, 1, 0] = .3333 * (
                    seed_img[:, :, 0, 0] + seed_img[:, :, 2, 0] + seed_img[:, :, 1, 1]) + torch.rand(
                1, device=device) * roughness - roughness / 2
            seed_img[:, :, 2, 1] = .3333 * (
                    seed_img[:, :, 2, 0] + seed_img[:, :, 2, 2] + seed_img[:, :, 1, 1]) + torch.rand(
                1, device=device) * roughness - roughness / 2
            seed_img[:, :, 1, 2] = .3333 * (
                    seed_img[:, :, 0, 2] + seed_img[:, :, 2, 2] + seed_img[:, :, 1, 1]) + torch.rand(
                1, device=device) * roughness - roughness / 2
            rnd_range = rnd_range * roughness
            recursion_depth -= 1
        else:
            while len(seed_img.size()) < 4:
                seed_img = seed_img.unsqueeze(dim=0)
            assert len(seed_img.size()) == 4
            assert seed_img.size(2) == seed_img.size(3) and 2 ** math.log2((seed_img.size(3) - 1)) + 1 == seed_img.size(
                3)
            seed_img = seed_img.to(device)
            rnd_range = 1.0 * roughness
        img = seed_img
        for n in range(1, recursion_depth + 1):
            img = one_diamond_one_square(img, rnd_range, random=random)
            rnd_range *= roughness
        if output_range is not None:
            img_max = img.max(dim=2)[0].max(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=2)
            img_min = img.min(dim=2)[0].min(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=2)
            img = (img - img_min) / (_epsilon + img_max - img_min)
            img = img * (output_range[1] - output_range[0]) + output_range[0]
        elif output_deviation_mean is not None:
            # standarizing image
            img = img - img.mean(dim=2).mean(dim=2).unsqueeze(dim=2).unsqueeze(dim=2)
            img = img / (img.std(dim=2).std(dim=2).unsqueeze(dim=2).unsqueeze(dim=2) + _epsilon)
            # setting mean and deviation
            img = img * (output_deviation_mean[0]) + output_deviation_mean[1]

        if width_height is not None:
            # TODO(anguelos) make sure there is no bias by cropping on the to left bias
            img = img[:, :, :width_height[0], :width_height[1]]
        if return_2d:
            assert replicates == 1
            return img[0, 0, :, :]
        else:
            return img


def _diamond_square_seed(replicates, width, height, random, device):
    assert width == 3 or height == 3
    if height == 3:
        transpose = True
        width, height = height, width
    else:
        transpose = False
    # width is always 3
    assert height % 2 == 1 and height > 2
    res = random([replicates, 1, width, height])
    res[:, :, ::2, ::2] = random([replicates, 1, 2, (height + 1)//2])
    # Diamond step
    res[:, :, 1, 1::2] = (res[:, :, ::2, :-2:2] + res[:, :, ::2, 2::2]).sum(dim=2) / 4.0
    # Square step
    if width > 3:
        res[:, :, 1, 2:-3:2] = (res[:, :, 0, 2:-3:2] + res[:, :, 2, 2:-3:2] + res[:, :, 1, 0:-4:2] + res[:, :, 1, 2:-3:2]) / 4.0

    res[:, :, 1, 0] = (res[:, :, 0, 0] + res[:, :, 1, 1] + res[:, :, 2, 0]) / 3.0
    res[:, :, 1, -1] = (res[:, :, -1, -1] + res[:, :, 1, -2] + res[:, :, 2, 0]) / 3.0
    res[:, :, 0, 1::2] = (res[:, :, 0, 0:-2:2] + res[:, :, 0, 2::2] + res[:, :, 1, 1::2]) / 3.0
    res[:, :, 2, 1::2] = (res[:, :, 2, 0:-2:2] + res[:, :, 2, 2::2] + res[:, :, 1, 1::2]) / 3.0
    if device is not None:
        res = res.to(device)
    if transpose:
        return res.transpose(2, 3)
    else:
        return res


# noinspection PyPackageRequirements
def functional_diamond_square(output_size, roughness=.5, rnd_scale=1.0, device=None,
                              seed_img=None, random=torch.rand, normalize_range=(0., 1.)) -> torch.Tensor:
    """Generates Plasma Fractal Images

    Args:
        width_height: A tuple of integers with the width and height of the image to be generated.
        replicates: An integer indicating how may diamond squares to compute in total.
        output_range: A tuple of floats indicating the range to which the fractals will be normalised. Must be None if
            output_deviation_mean is not None. If None
        recursion_depth:
        roughness:
        device:
        seed_img:
        return_2d:
        output_deviation_mean:

    Returns:

    """
    if not isinstance(rnd_scale, torch.Tensor):
        rnd_scale = torch.Tensor([rnd_scale]).view(1, 1, 1, 1).to(device)
        rnd_scale = rnd_scale.expand([output_size[0] * output_size[1], 1, 1, 1])
    else:
        rnd_scale = rnd_scale.view(-1, 1, 1, 1)
        rnd_scale = rnd_scale.expand([output_size[0],  output_size[1], 1, 1])
        assert rnd_scale.size(0) == output_size[0]
        rnd_scale = rnd_scale.reshape([-1, 1, 1, 1])
    if not isinstance(roughness, torch.Tensor):
        roughness = torch.Tensor([roughness]).view(1, 1, 1, 1)
        roughness = roughness.expand([output_size[0] * output_size[1], 1, 1, 1])
    else:
        roughness = roughness.view(-1, 1, 1, 1)
        roughness = roughness.expand([output_size[0],  output_size[1], 1, 1])
        assert roughness.size(0) == output_size[0]
        roughness = roughness.reshape([-1, 1, 1, 1])
    if device is not None:
        roughness = roughness.to(device)
        rnd_scale = rnd_scale.to(device)
    width, height = tuple(output_size)[-2:]
    n_samples = 1
    for dim_size in tuple(output_size)[:-2]:
        n_samples *= dim_size
    if seed_img is None:
        p2_width = 2 ** math.ceil(math.log2(width - 1)) + 1
        p2_height = 2 ** math.ceil(math.log2(height - 1)) + 1
        recursion_depth = int(min(math.log2(p2_width - 1) - 1, math.log2(p2_height - 1) - 1))
        seed_width = (p2_width - 1) // 2 ** recursion_depth + 1
        seed_height = (p2_height - 1) // 2 ** recursion_depth + 1
        img = _diamond_square_seed(replicates=n_samples, width=seed_width, height=seed_height, random=random,
                                   device=device) * rnd_scale
    else:
        raise NotImplemented()
    for _ in range(recursion_depth):
        rnd_scale = rnd_scale * roughness
        img = one_diamond_one_square(img, rnd_scale, random=random)

    img = img[:, :, :width, :height]
    img = img.view(output_size)
    if normalize_range is not None:
        sample_min = img.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].view([-1, 1, 1, 1])
        sample_max = img.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].view([-1, 1, 1, 1])
        sample_range = sample_max - sample_min
        img = (img - sample_min) / sample_range
    return img


default_diamond_kernel = [[.25, 0., .25], [0., 0., 0.], [.25, 0., .25]]
default_diamond_kernel = torch.tensor(default_diamond_kernel).unsqueeze(dim=0).unsqueeze(dim=0)
default_square_kernel = [[0., .25, 0.], [.25, 0., .25], [0., .25, 0.]]
default_square_kernel = torch.tensor(default_square_kernel).unsqueeze(dim=0).unsqueeze(dim=0)


def one_diamond_one_square(img, rnd_scale, random=torch.rand, diamond_kernel=default_diamond_kernel, square_kernel=default_square_kernel):
    """Doubles the image resolution by applying a single diamond square steps

    Recursive application of this method creates plasma fractals.
    Attention! The function is differentiable and gradients are computed as well.
    If this function is run in the usual sense, it is more efficient if it is run in a no_grad()

    :param img: A 4D tensor where dimensions are Batch, Channel, Width, Height. Width and Height must both be 2^N+1 and
        Batch and Channels should in the usual case be 1.
    :param rnd_scale: A float  number in [0,1] controlling the randomness created pixels get. I the usual case, it is
        halved at every applycation of this function.
    :return: A tensor on the same device as img with the same channels as img and width, height of 2^(N+1)+1
    """
    # TODO (anguelos) test multi channel and batch size > 1

    batch_sz, _, _, _ = img.size()
    step = 2
    new_img = torch.zeros([batch_sz, 1, 2 * (img.shape[2] - 1) + 1, 2 * (img.shape[3] - 1) + 1], device=img.device)
    new_img[:, :, ::step, ::step] = img

    pad_compencate = torch.ones_like(new_img)
    pad_compencate[:, :, :, 0] = 1 / .75
    pad_compencate[:, :, :, -1] = 1 / .75
    pad_compencate[:, :, 0, :] = 1 / .75
    pad_compencate[:, :, -1, :] = 1 / .75

    rnd_img = random(new_img.size(), device=img.device) * rnd_scale

    # diamond
    diamond_regions = torch.nn.functional.conv2d(new_img, diamond_kernel.to(img.device), padding=1)
    diamond_centers = (diamond_regions > 0).float()
    # TODO (anguelos) make sure diamond_regions*diamond_centers is needed
    new_img = new_img + (1 - rnd_scale) * diamond_regions * diamond_centers + diamond_centers * rnd_img

    # square
    square_regions = torch.nn.functional.conv2d(new_img, square_kernel.to(img.device), padding=1) * pad_compencate
    square_centers = (square_regions > 0).float()
    # TODO (anguelos) make sure square_centers*square_regions is needed
    new_img = new_img + square_centers * rnd_img + (1 - rnd_scale) * square_centers * square_regions

    return new_img


class DiamondSquare(torch.nn.Module):
    def get_current_device(self):
        return self.initial_rnd_scale.data.device

    def __init__(self, recursion_steps=1, rnd_scale=1.0, rand=torch.rand):
        self.recursion_steps = recursion_steps
        self.rand = rand
        self.initial_rnd_range = torch.nn.Parameter(torch.tensor([rnd_scale], requires_grad=True))
        self.diamond_kernel = [[.25, 0., .25], [0., 0., 0.], [.25, 0., .25]]
        self.diamond_kernel = torch.nn.Parameter(torch.tensor(
            self.diamond_kernel, requires_grad=True).unsqueeze(dim=0).unsqueeze(dim=0))
        self.square_kernel = [[0., .25, 0.], [.25, 0., .25], [0., .25, 0.]]
        self.square_kernel = torch.nn.Parameter(torch.tensor(
            self.square_kernel, requires_grad=True).unsqueeze(dim=0).unsqueeze(dim=0))
        self.initial_rnd_scale=torch.nn.Parameter(torch.tensor(rnd_scale,requires_grad=False))

    def forward(self, input_img=None, seed_size=None):
        if input is None:
            img = _diamond_square_seed(seed_size, device=self.get_current_device())
        else:
            img = input_img
        rnd_scale = self.initial_rnd_range.clone()
        for _ in range(self.recursion_steps):
            img = one_diamond_one_square(img, random=self.rand, diamond_kernel=self.diamond_kernel, square_kernel=self.square_kernel, rnd_scale=rnd_scale)
        return img

