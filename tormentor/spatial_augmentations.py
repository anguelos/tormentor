from .base_augmentation import SpatialImageAugmentation
from .backgrounds import *

import torch
import torch.nn.functional as F
import kornia
import math


# Simple example of using the augmentatio n API
class Rotate(SpatialImageAugmentation):
    def forward(self, tensor_image):
        self.rotate_radians = (torch.rand([1]) + self.min_angle)*(self.max_angle-self.min_angle)
        # TODO (anguelos) is there a better way than self. to store angles or other variables defining the augmentation?
        rotate_degrees = (self.rotate_radians * 180) / math.pi
        return kornia.geometry.rotate(tensor_image,rotate_degrees)

    @classmethod
    def factory(cls, min_angle=-math.pi, max_angle=math.pi):
        return lambda: cls(min_angle=min_angle, max_angle=max_angle)


class Scale(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images

    """
    def forward(self, tensor_image):
        if self.joint:
            scale = (torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale).item()
            result = F.interpolate(tensor_image, scale_factor=[scale, scale], mode='bilinear', align_corners=True)
        else:
            width_height_ratios = (torch.rand(2) * (self.max_scale - self.min_scale) + self.min_scale)
            result = F.interpolate(tensor_image, scale_factor=width_height_ratios, mode='bilinear', align_corners=True)
        return result

    @classmethod
    def factory(cls, min_scale=.5, max_scale=2.0, joint=False):
        return lambda: cls(min_scale=min_scale, max_scale=max_scale, joint=joint)


class Flip(SpatialImageAugmentation):
    def forward(self, tensor_image):
        dim_thresholds = torch.Tensor([1.0, 1.0, self.horizontal_prob, self.vertical_prob])
        rnd = torch.rand(4)
        flip_dims = torch.nonzero(rnd > dim_thresholds).view(-1)
        flip_dims = tuple(flip_dims.tolist())  # TODO (anguelos) can we use a tensor instead of a list
        return tensor_image.flip(dims=flip_dims)

    @classmethod
    def factory(cls, horizontal_prob=.5, vertical_prob=.5):
        return lambda: cls(horizontal_prob=horizontal_prob, vertical_prob=vertical_prob)


class EraseRectangle(SpatialImageAugmentation):
    def forward(self, tensor_image):
        _, _, image_width, image_height = tensor_image.size()

        size_minima = torch.Tensor([self.min_width, self.min_height])
        size_ranges = torch.Tensor([self.max_width - self.min_width, self.max_height - self.min_height])
        patch_width, patch_height = (torch.rand(2) * size_ranges + size_minima).tolist()
        patch_width
        flip_dims = tuple(flip_dims.tolist())  # TODO (anguelos) can we use a tensor instead of a list
        return tensor_image.flip(dims=flip_dims)

    @classmethod
    def factory(cls, min_width=.1, max_width=.5, min_height=.1, max_height=.5):
        return lambda: cls(min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height)


class CropPadAsNeeded(SpatialImageAugmentation):
    def forward(self, tensor_image):
        _, nb_channels, input_width, input_height = tensor_image.size()
        new_tensor_image = torch.zeros([1, nb_channels, self.desired_width, self.desired_height])

        if input_width > self.desired_width:
            input_left = torch.randint(0, input_width - self.desired_width, (1,)).item()
            input_right = input_left + self.desired_width
            output_left = 0
            output_right = self.desired_width
        elif input_width < self.desired_width:
            input_left = 0
            input_right = input_width
            output_left = torch.randint(0, self.desired_width - input_width, (1,)).item()
            output_right = output_left + input_width
        else: # They are equal
            input_left = output_left = 0
            input_right = output_right = self.desired_width

        if input_height > self.desired_height:
            input_top = torch.randint(0, input_height - self.desired_height, (1,)).item()
            input_bottom = input_top + self.desired_height
            output_top = 0
            output_bottom = self.desired_height
        elif input_height < self.desired_height:
            input_top = 0
            input_bottom = input_height
            output_top = torch.randint(0, self.desired_height - input_height, (1,)).item()
            output_bottom = output_top + input_height
        else:  # They are equal
            input_top = output_top = 0
            input_bottom = output_bottom = self.desired_width

        new_tensor_image[:, :, output_left:output_right, output_top:output_bottom] = \
            tensor_image[:, :, input_left:input_right, input_top:input_bottom]

        return new_tensor_image

    @classmethod
    def factory(cls, desired_width, desired_height):
        return lambda: cls(desired_width=desired_width, desired_height=desired_height)


class ScaleAndPadAsNeeded(SpatialImageAugmentation):
    def forward(self, tensor_image):
        _, nb_channels, input_width, input_height = tensor_image.size()
        if self.preserve_aspect_ratio:
            new_tensor_image = torch.zeros(1, nb_channels, self.desired_width, self.desired_height)
            if input_width/input_height > self.desired_width/self.desired_height: # scaling to desired width
                scaled_heigth = int(round(self.desired_height * (input_height/ input_width)))
                scaled_tensor_image = new_tensor_image = F.interpolate(tensor_image,
                                                                       size=[self.desired_width, scaled_heigth],
                                                                       mode='bilinear', align_corners=True)
                top = torch.randint(0, scaled_heigth - self.desired_height, (1,)).item()
                bottom = self.desired_height - top
                new_tensor_image[:,:,:,top:bottom] = scaled_tensor_image
            else: #scaling to desired width
                scaled_width = int(round(self.desired_width * (input_width/input_height)))
                scaled_tensor_image = new_tensor_image = F.interpolate(tensor_image,
                                                                       size=[scaled_width, self.desired_height],
                                                                       mode='bilinear', align_corners=True)
                print(scaled_width, self.desired_width)
                if scaled_width == self.desired_width:
                    left = 0
                else:
                    left = torch.randint(0, scaled_width - self.desired_width, (1,)).item()
                right = self.desired_width - left
                new_tensor_image[:,:, left:right, :] = scaled_tensor_image
        else:
            new_tensor_image = F.interpolate(tensor_image, size=[self.desired_width, self.desired_width],
                                             mode='bilinear', align_corners=True)
        return new_tensor_image

    @classmethod
    def factory(cls, desired_width=128, desired_height=128, preserve_aspect_ratio=True):
        return lambda: cls(desired_width=desired_width, desired_height=desired_height,
                           preserve_aspect_ratio=preserve_aspect_ratio)
