import kornia as K
import torch

from .base_augmentation import SpatialImageAugmentation, SamplingField, AugmentationState
from .random import Uniform, Bernoulli


class Perspective(SpatialImageAugmentation):
    r"""Applies a perspective transformation on the data by moving the corners of an image.

   This augmentation is parametrised by two random variables ``x_offset`` and ``y_offset`` which are the multipliers of
   each of the image corners corners (-1, -1), (1, -1), (1, 1), and (-1, 1).

   .. image:: _static/example_images/Perspective.png
   """

    x_offset = Uniform((.75, 1.5))
    y_offset = Uniform((.75, 1.5))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> AugmentationState:
        batch_sz = sampling_tensors[0].size(0)
        top_left_x = -1 * type(self).x_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        top_right_x = 1 * type(self).x_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        bottom_left_x = -1 * type(self).x_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        bottom_right_x = 1 * type(self).x_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        top_left_y = -1 * type(self).y_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        top_right_y = -1 * type(self).y_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        bottom_left_y = 1 * type(self).y_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        bottom_right_y = 1 * type(self).y_offset(batch_sz, device=sampling_tensors[0].device).view(-1, 1, 1)
        dst_x = torch.cat([top_left_x, top_right_x, bottom_left_x, bottom_right_x], dim=1)
        dst_y = torch.cat([top_left_y, top_right_y, bottom_left_y, bottom_right_y], dim=1)
        dst_y = torch.cat([top_left_y, top_right_y, bottom_left_y, bottom_right_y], dim=1)
        dst = torch.cat([dst_x, dst_y], dim=2)
        src = torch.ones_like(dst)
        src[:, [0, 2], 0] = -1
        src[:, [0, 1], 1] = -1
        return K.geometry.transform.get_perspective_transform(src, dst),

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, affine_matrices) -> SamplingField:
        X, Y = sampling_field
        Z = torch.ones_like(X)
        new_X = affine_matrices[:, 0:1, 0:1] * X + affine_matrices[:, 1:2, 0:1] * Y + affine_matrices[:, 2:3, 0:1] * Z
        new_Y = affine_matrices[:, 0:1, 1:2] * X + affine_matrices[:, 1:2, 1:2] * Y + affine_matrices[:, 2:3, 1:2] * Z
        new_Z = affine_matrices[:, 0:1, 2:3] * X + affine_matrices[:, 1:2, 2:3] * Y + affine_matrices[:, 2:3, 2:3] * Z
        new_X = new_X / new_Z
        new_Y = new_Y / new_Z
        return new_X, new_Y


class Rotate(SpatialImageAugmentation):
    r"""Rotates the image around the center.

   .. image:: _static/example_images/Rotate.png
   """
    radians = Uniform((-3.1415, 3.1415))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> AugmentationState:
        batch_sz = sampling_tensors[0].size(0)
        radians = type(self).radians(batch_sz, device=sampling_tensors[0].device).view(-1)
        return radians,

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, radians: torch.FloatTensor) -> SamplingField:
        field_x, field_y = sampling_field
        radians = radians.unsqueeze(dim=1).unsqueeze(dim=1)
        cos_th = torch.cos(radians)
        sin_th = torch.sin(radians)
        neg_sin_th = torch.sin(-radians)
        field_x, field_y = field_x * cos_th + neg_sin_th * field_y, field_x * sin_th + cos_th * field_y
        return field_x, field_y

    @classmethod
    def inverse_functional_sampling_field(cls, sampling_field: SamplingField,
                                          radians: torch.FloatTensor) -> SamplingField:
        field_x, field_y = sampling_field
        radians = -radians.unsqueeze(dim=1).unsqueeze(dim=1)
        cos_th = torch.cos(radians)
        sin_th = torch.sin(radians)
        neg_sin_th = torch.sin(-radians)
        field_x, field_y = field_x * cos_th + neg_sin_th * field_y, field_x * sin_th + cos_th * field_y
        return field_x, field_y


class Zoom(SpatialImageAugmentation):
    r"""Augments by scaling images preserving their aspect ratio.

   .. image:: _static/example_images/Zoom.png
   """
    scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_field: SamplingField) -> AugmentationState:
        scales = type(self).scales(sampling_field[0].size(0), device=sampling_field[0].device)
        return scales,

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, scales: torch.FloatTensor) -> SamplingField:
        scales = scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return sampling_field[0] / scales, sampling_field[1] / scales


class Scale(SpatialImageAugmentation):
    r"""Augmentation by scaling images preserving aspect ratio.

   .. image:: _static/example_images/Scale.png
   """
    x_scales = Uniform(value_range=(.5, 1.5))
    y_scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_scales = type(self).x_scales(batch_sz, device=sampling_tensors[0].device)
        y_scales = type(self).y_scales(batch_sz, device=sampling_tensors[0].device)
        return (x_scales, y_scales)

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, x_scales: torch.FloatTensor,
                                  y_scales: torch.FloatTensor):
        x_scales = x_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        y_scales = y_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return x_scales * sampling_field[0], y_scales * sampling_field[1]


class Translate(SpatialImageAugmentation):
    r"""Augmentation by translating images.

   .. image:: _static/example_images/Translate.png
   """
    x_offset = Uniform(value_range=(-1., 1.))
    y_offset = Uniform(value_range=(-1., 1.))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_offset = type(self).x_offset(batch_sz, device=sampling_tensors[0].device)
        y_offset = type(self).y_offset(batch_sz, device=sampling_tensors[0].device)
        return (x_offset, y_offset)

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, x_offset: torch.FloatTensor,
                                  y_offset: torch.FloatTensor):
        x_offset = x_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        y_offset = y_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        return x_offset + sampling_field[0], y_offset + sampling_field[1]


class ScaleTranslate(SpatialImageAugmentation):
    r"""Augmentation by scaling and translating images preserving aspect ratio.

   .. image:: _static/example_images/ScaleTranslate.png
   """
    x_offset = Uniform(value_range=(-1., 1.))
    y_offset = Uniform(value_range=(-1., 1.))
    x_scales = Uniform(value_range=(.5, 1.5))
    y_scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_offset = type(self).x_offset(batch_sz, device=sampling_tensors[0].device)
        y_offset = type(self).y_offset(batch_sz, device=sampling_tensors[0].device)
        x_scales = type(self).x_scales(batch_sz, device=sampling_tensors[0].device)
        y_scales = type(self).y_scales(batch_sz, device=sampling_tensors[0].device)
        return (x_offset, y_offset, x_scales, y_scales)

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, x_offset: torch.FloatTensor,
                                  y_offset: torch.FloatTensor, x_scales: torch.FloatTensor,
                                  y_scales: torch.FloatTensor):
        x_offset = x_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        y_offset = y_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        x_scales = x_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        y_scales = y_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return x_offset + x_scales * sampling_field[0], y_offset + y_scales * sampling_field[1]


class Flip(SpatialImageAugmentation):
    r"""Implementation of augmentation by flipping the X or Y axis.

   .. image:: _static/example_images/Flip.png
   """

    horizontal = Bernoulli(.5)
    vertical = Bernoulli(.5)

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        horizontal = type(self).horizontal(batch_sz, device=sampling_tensors[0].device)
        vertical = type(self).vertical(batch_sz, device=sampling_tensors[0].device)
        return horizontal, vertical

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, horizontal: torch.FloatTensor,
                                  vertical: torch.FloatTensor):
        horizontal = ((1 - horizontal) * 2 - 1).unsqueeze(dim=1).unsqueeze(dim=1)
        vertical = ((1 - vertical) * 2 - 1).unsqueeze(dim=1).unsqueeze(dim=1)
        return horizontal * sampling_field[0], vertical * sampling_field[1]


class RemoveRectangle(SpatialImageAugmentation):
    r"""Augmentation EraseRectangle.

    .. image:: _static/example_images/RemoveRectangle.png
    """
    center_x = Uniform((-1.0, 1.0))
    center_y = Uniform((-1.0, 1.0))
    width = Uniform((.2, .5))
    height = Uniform((.2, .5))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        batch_size = sampling_tensors[0].size(0)
        center_x = type(self).center_x(batch_size, device=sampling_tensors[0].device)
        center_y = type(self).center_y(batch_size, device=sampling_tensors[0].device)
        width = type(self).width(batch_size, device=sampling_tensors[0].device)
        height = type(self).height(batch_size, device=sampling_tensors[0].device)
        return center_x, center_y, width, height

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, center_x: torch.FloatTensor,
                                  center_y: torch.FloatTensor, width: torch.FloatTensor,
                                  height: torch.FloatTensor) -> SamplingField:
        # TODO(anguelos) make pushing the rectangle to its nearest edge instead of corner

        center_x = center_x.view(-1, 1, 1)
        center_y = center_y.view(-1, 1, 1)
        width = width.view(-1, 1, 1)
        height = height.view(-1, 1, 1)

        left = center_x - width / 2
        right = center_x + width / 2
        top = center_y - height / 2
        bottom = center_y + height / 2

        X, Y = sampling_field

        left_half = (X > left) * (X < center_x)
        top_half = (Y > top) * (Y < center_y)

        right_half = (X < right) * (X >= center_x)
        bottom_half = (Y < bottom) * (Y >= center_y)

        X = X - left_half * X + left_half * left - right_half * X + right_half * right
        Y = Y - top_half * Y + top_half * left - bottom_half * Y + bottom_half * right
        return X, Y


class ElasticTransform(SpatialImageAugmentation):
    r"""Augmentation ElasticTransform.

    This augmentation does not guaranty to be a homomorphism. In order for the augmentation to behave as a homomorphism,
    ``harmonic_smoothing`` must be quite low. On the other hand, the complexity of the operation is n**2 with respect to
    ``harmonic_smoothing`` or n log(n) depending of how the gaussian filters are implemented.

    .. image:: _static/example_images/ElasticTransform.png
    """
    offset_x = Uniform((-1.0, 1.0))
    offset_y = Uniform((-1.0, 1.0))
    harmonic_smoothing = Uniform((10, 100))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        X, Y = sampling_tensors
        offset_x = type(self).offset_x(X.size())
        offset_y = type(self).offset_y(Y.size())
        harmonic_smoothing = type(self).harmonic_smoothing(X.size(0)).view([-1, 1, 1])
        x_variances = (1 / harmonic_smoothing) * X.size(2) / 2
        y_variances = (1 / harmonic_smoothing) * X.size(1) / 2
        x_variances = torch.clamp(x_variances, 3., X.size(2))
        y_variances = torch.clamp(y_variances, 3., X.size(1))
        x_filter_sizes = (x_variances.ceil().long() * 2 + 1)
        y_filter_sizes = (y_variances.ceil().long() * 2 + 1)
        dX = torch.empty_like(X)
        dY = torch.empty_like(Y)
        for n in range(X.size(0)):
            # sigma, filter_size = variances[n].item(), filter_sizes[n].item()
            sigma = x_variances[n].item(), y_variances[n].item()
            filter_size = x_filter_sizes[n].item(), y_filter_sizes[n].item()
            sample_offset_x = offset_x[n: n + 1, :, :].unsqueeze(dim=1)
            sample_offset_y = offset_y[n: n + 1, :, :].unsqueeze(dim=1)
            sample_offset_x = K.filters.gaussian_blur2d(sample_offset_x, kernel_size=filter_size, sigma=sigma)
            sample_offset_y = K.filters.gaussian_blur2d(sample_offset_y, kernel_size=filter_size, sigma=sigma)
            dX[n, :, :] = sample_offset_x[0, 0, :, :]
            dY[n, :, :] = sample_offset_y[0, 0, :, :]
        return dX, dY

    @classmethod
    def functional_sampling_field(cls, sampling_field: SamplingField, dX: torch.FloatTensor,
                                  dY: torch.FloatTensor) -> SamplingField:
        X, Y = sampling_field
        return X + dX, Y + dY
