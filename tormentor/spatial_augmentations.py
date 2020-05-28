from .base_augmentation import SpatialImageAugmentation, SamplingField, SpatialAugmentationState
from .random import Uniform, Bernoulli, Categorical
import torch


class Rotate(SpatialImageAugmentation):
    rotate_radians = Uniform((-3.1415, 3.1415))

    def generate_batch_state(self, sampling_tensors:SamplingField)->SpatialAugmentationState:
        batch_sz = sampling_tensors[0].size(0)
        radians = type(self).rotate_radians(batch_sz).view(-1)
        return (radians,)

    @staticmethod
    def functional_points(sampling_field:SamplingField, radians:torch.FloatTensor)->SamplingField:
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

    def generate_batch_state(self, sampling_tensors:SamplingField)->torch.FloatTensor:
        scales = type(self).scales(sampling_tensors[0].size(0))
        return (scales,)

    @staticmethod
    def functional_points(sampling_field:SamplingField, scales:torch.FloatTensor):
        scales = scales.unsqueeze(dim=1).unsqueeze(dim=2)
        return scales * sampling_field[0], scales * sampling_field[1]


class Scale(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images preserving aspect ratio.
    """
    x_scales = Uniform(value_range=(.5, 1.5))
    y_scales = Uniform(value_range=(.5, 1.5))

    def generate_batch_state(self, sampling_tensors:SamplingField)->torch.FloatTensor:
        batch_sz = sampling_tensors[0].size(0)
        x_scales = type(self).x_scales(batch_sz)
        y_scales = type(self).y_scales(batch_sz)
        return (x_scales, y_scales)

    @staticmethod
    def functional_points(sampling_field:SamplingField, x_scales:torch.FloatTensor, y_scales:torch.FloatTensor):
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
        x_offset = type(self).x_offset(batch_sz)
        y_offset = type(self).y_offset(batch_sz)
        return (x_offset, y_offset)

    @staticmethod
    def functional_points(sampling_field:SamplingField, x_offset:torch.FloatTensor, y_offset:torch.FloatTensor):
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
        x_offset = type(self).x_offset(batch_sz)
        y_offset = type(self).y_offset(batch_sz)
        x_scales = type(self).x_scales(batch_sz)
        y_scales = type(self).y_scales(batch_sz)
        return (x_offset, y_offset, x_scales, y_scales)

    @staticmethod
    def functional_points(sampling_field:SamplingField, x_offset:torch.FloatTensor, y_offset:torch.FloatTensor, x_scales:torch.FloatTensor, y_scales:torch.FloatTensor):
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
        horizontal, vertical = type(self).horizontal(batch_sz), type(self).vertical(batch_sz)
        return horizontal, vertical

    @staticmethod
    def functional_points(sampling_field:SamplingField, horizontal:torch.FloatTensor, vertical:torch.FloatTensor):
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
        center_x = type(self).center_x(batch_size)
        center_y = type(self).center_y(batch_size)
        width = type(self).width(batch_size)
        height = type(self).height(batch_size)
        return center_x, center_y, width, height

    @staticmethod
    def functional_points(sampling_field:SamplingField, center_x:torch.FloatTensor, center_y:torch.FloatTensor, width:torch.FloatTensor, height:torch.FloatTensor)->SamplingField:
        center_x = center_x.view(-1, 1, 1)
        center_y = center_y.view(-1, 1, 1)
        width = width.view(-1, 1, 1)
        height = height.view(-1, 1, 1)
        x_sampling, y_sampling = sampling_field
        x_erase = ((x_sampling - center_x).abs() < width).float() * EraseRectangle.outside_field
        y_erase = ((y_sampling - center_y).abs() < height).float() * EraseRectangle.outside_field
        erase = x_erase * y_erase
        sampling_field = x_sampling + erase, y_sampling + erase
        return sampling_field
