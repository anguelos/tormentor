import torch
import kornia as K
from typing import Tuple

SamplingField = Tuple[torch.FloatTensor, torch.FloatTensor]


def _use_sampling_field(sf: SamplingField):
    return None

def is_sampling_field(var):
    r"""Returns True is var is a SamplingField.

    A tuple containing two torch.FloatTensors

    Args:
        var:

    Returns:

    """
    raise NotImplementedError
    #try:
    #    _use_sampling_field(var)
    #    return True
    #except TypeError:
    #    return False


def create_sampling_field(width: int, height: int, batch_size: int = 1, device: torch.device = "cpu") -> SamplingField:
    r"""Creates a SamplingField.

    A SamplingField is a tuple of 3D tensors of the same size. Sampling fields are augmentable by all augmentations
    although many augmentations (Non-spatial) have no effect on them. The can be used to resample images, pointclouds,
    masks. When sampling, for both axes, the input image is interpreted to lie on the region [-1, 1]. The output image
    when resampling will have the width and height of the sampling field. A sampling field can also refer to a single
    image rather than a batch in whitch case the tensors are 2D.
    The first dimension is the batch size.
    The second dimension is the width of the output image after sampling.
    The third dimension is the width of the output image after sampling.
    The created sampling fields are normalised in the range [-1,1] regardless of their size.
    Although not enforced, it is expected that augmentations are homomorphisms.
    Sampling fields are expected to operate identically on all channels and dont have a channel dimension.

    Args:
        width: The sampling fields width.
        height:  The sampling fields height.
        batch_size: If 0, the sampling field refers to a single image. Otherwise the first dimension of the tensors.
            Created sampling fileds are simply repeated over the batch dimension. Default value is 1.
        device: the device on which the sampling filed will be created.

    Returns:
        A tuple of 3D or 2D tensors with values ranged in [-1,1]

    """
    sf = K.utils.create_meshgrid(height=height, width=width, normalized_coordinates=True, device=device)
    sf = (sf[:, :, :, 0], sf[:, :, :, 1])
    if batch_size == 0:
        return sf[0][0, :, :], sf[1][0, :, :]
    else:
        return sf[0].repeat([batch_size, 1, 1]), sf[1].repeat([batch_size, 1, 1])


def apply_sampling_field(input_img: torch.Tensor, coords: SamplingField):
    r"""Resamples one or more images by applying sampling fields.

    Bilinear interpolation is employed.

    Args:
        input_img: A 4D float tensor [batch x channel x height x width] or a 3D tensor [channel x height x width].
            Containing the image or batch from which the image is sampled.
        coords: A sampling field with 3D [batch x out_height x out_width] or 2D [out_height x out_width]. The dimensions
            of the sampling fields must be one less that the input_img.

    Returns:
        A tensor of as many dimensions [batch x channel x out_height x out_width] or [channel x out_height x out_width]]
        as the input.
    """
    x_coords, y_coords = coords
    if input_img.ndim == 3:
        assert coords[0].ndim == 2
        x_coords, y_coords = x_coords.unsqueeze(dim=0), y_coords.unsqueeze(dim=0)
        batch = input_img.unsqueeze(dim=0)
    else:
        batch = input_img
    xy_coords = torch.cat((x_coords.unsqueeze(dim=-1), y_coords.unsqueeze(dim=-1)), dim=3)
    sampled_batch = torch.nn.functional.grid_sample(batch, xy_coords, align_corners=True)
    if input_img.ndim == 3:
        return sampled_batch[0, :, :, :]
    else:
        return sampled_batch
