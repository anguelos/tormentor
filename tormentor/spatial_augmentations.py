from .base_augmentation import SpatialImageAugmentation, SamplingField, SpatialAugmentationState
from .random import Uniform, Bernoulli, Categorical
import torch
import kornia as K


class Rotate(SpatialImageAugmentation):
    radians = Uniform((-3.1415, 3.1415))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> SpatialAugmentationState:
        batch_sz = sampling_tensors[0].size(0)
        radians = type(self).radians(batch_sz, device=sampling_tensors[0].device).view(-1)
        print("Radians:", radians)
        return (radians,)

    @staticmethod
    def functional_sampling_field(sampling_field: SamplingField, radians:torch.FloatTensor) -> SamplingField:
        field_x, field_y = sampling_field
        radians = radians.unsqueeze(dim=1).unsqueeze(dim=1)
        cos_th = torch.cos(radians)
        sin_th = torch.sin(radians)
        neg_sin_th = torch.sin(-radians)
        return field_x * cos_th + neg_sin_th * field_y, field_x*sin_th + cos_th*field_y


class Zoom(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images preserving aspect ratio.
    """
    scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> SpatialAugmentationState:
        scales = type(self).scales(sampling_tensors[0].size(0), device=sampling_tensors[0].device)
        return (scales,)

    @staticmethod
    def functional_sampling_field(sampling_field: SamplingField, scales:torch.FloatTensor) -> SamplingField:
        scales = scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return sampling_field[0] / scales, sampling_field[1] / scales


class Scale(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images preserving aspect ratio.
    """
    x_scales = Uniform(value_range=(.5, 1.5))
    y_scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_tensors:SamplingField)->torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_scales = type(self).x_scales(batch_sz, device=sampling_tensors[0].device)
        y_scales = type(self).y_scales(batch_sz, device=sampling_tensors[0].device)
        return (x_scales, y_scales)

    @staticmethod
    def functional_sampling_field(sampling_field:SamplingField, x_scales:torch.FloatTensor, y_scales:torch.FloatTensor):
        x_scales = x_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        y_scales = y_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return x_scales * sampling_field[0], y_scales * sampling_field[1]


class Translate(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images preserving aspect ratio.
    """
    x_offset = Uniform(value_range=(-1., 1.))
    y_offset = Uniform(value_range=(-1., 1.))

    def generate_batch_state(self, sampling_tensors:SamplingField)->torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_offset = type(self).x_offset(batch_sz, device=sampling_tensors[0].device)
        y_offset = type(self).y_offset(batch_sz, device=sampling_tensors[0].device)
        return (x_offset, y_offset)

    @staticmethod
    def functional_sampling_field(sampling_field:SamplingField, x_offset:torch.FloatTensor, y_offset:torch.FloatTensor):
        x_offset = x_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        y_offset = y_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        return x_offset + sampling_field[0], y_offset + sampling_field[1]


class ScaleTranslate(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images preserving aspect ratio.
    """
    x_offset = Uniform(value_range=(-1., 1.))
    y_offset = Uniform(value_range=(-1., 1.))
    x_scales = Uniform(value_range=(.5, 1.5))
    y_scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_tensors:SamplingField)->torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_offset = type(self).x_offset(batch_sz, device=sampling_tensors[0].device)
        y_offset = type(self).y_offset(batch_sz, device=sampling_tensors[0].device)
        x_scales = type(self).x_scales(batch_sz, device=sampling_tensors[0].device)
        y_scales = type(self).y_scales(batch_sz, device=sampling_tensors[0].device)
        return (x_offset, y_offset, x_scales, y_scales)

    @staticmethod
    def functional_sampling_field(sampling_field:SamplingField, x_offset:torch.FloatTensor, y_offset:torch.FloatTensor, x_scales:torch.FloatTensor, y_scales:torch.FloatTensor):
        x_offset = x_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        y_offset = y_offset.unsqueeze(dim=1).unsqueeze(dim=2)
        x_scales = x_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        y_scales = y_scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return x_offset + x_scales * sampling_field[0], y_offset + y_scales * sampling_field[1]


class Flip(SpatialImageAugmentation):
    horizontal = Bernoulli(.5)
    vertical = Bernoulli(.5)

    def generate_batch_state(self, sampling_tensors:SamplingField)->torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        horizontal = type(self).horizontal(batch_sz, device=sampling_tensors[0].device)
        vertical = type(self).vertical(batch_sz, device=sampling_tensors[0].device)
        return horizontal, vertical

    @staticmethod
    def functional_sampling_field(sampling_field:SamplingField, horizontal:torch.FloatTensor, vertical:torch.FloatTensor):
        horizontal = ((1-horizontal) * 2 - 1).unsqueeze(dim=1).unsqueeze(dim=1)
        vertical = ((1 - vertical) * 2 - 1).unsqueeze(dim=1).unsqueeze(dim=1)
        return horizontal * sampling_field[0], vertical * sampling_field[1]


class EraseRectangle(SpatialImageAugmentation):
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

    @staticmethod
    def functional_sampling_field(sampling_field:SamplingField, center_x:torch.FloatTensor, center_y:torch.FloatTensor, width:torch.FloatTensor, height:torch.FloatTensor)->SamplingField:
        """Erases a rectangle from a sampling field.

        It pushes all points inside the rectangle towards the nearest corner of this rectangle.

        Args:
            sampling_field ():
            center_x ():
            center_y ():
            width ():
            height ():

        Returns:

        """
        #TODO(anguelos) make pushing the rectangle to its nearest edge instead of corner

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
    offset_x = Uniform((-1.0, 1.0))
    offset_y = Uniform((-1.0, 1.0))
    harmonic_smoothing = Uniform((5,100))

    def generate_batch_state(self, sampling_tensors: SamplingField) -> torch.FloatTensor:
        X, Y = sampling_tensors
        offset_x = type(self).offset_x(X.size())
        offset_y = type(self).offset_y(Y.size())
        harmonic_smoothing = type(self).harmonic_smoothing(X.size(0)).view([-1,1,1])
        variances = (1/harmonic_smoothing) * X.size(1)
        filter_sizes = (variances*2+1).long()
        dX = torch.empty_like(X)
        dY = torch.empty_like(Y)
        for n in range(X.size(0)):
            sigma, filter_size = variances[n].item(), filter_sizes[n].item()
            dX[n, :, :] = K.filters.gaussian_blur2d(offset_x[n, :, :], (sigma, sigma), (filter_size, filter_size))
            dY[n, :, :] = K.filters.gaussian_blur2d(offset_y[n, :, :], (sigma, sigma), (filter_size, filter_size))
        return dX, dY

    @staticmethod
    def functional_sampling_field(self, sampling_field:SamplingField, dX:torch.FloatTensor, dY:torch.FloatTensor)->SamplingField:
        X, Y = sampling_field
        return X + dX, Y + dY
