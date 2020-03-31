from .base_augmentation import SpatialImageAugmentation, aug_parameters, aug_distributions
from .random import Uniform, Uniform2D, Bernoulli, Categorical

import torch
import torch.nn.functional as F
import kornia
import math


# Fully diferentiable augmentation
@aug_distributions(rotate_radians=Uniform((-3.1415, 3.1415)))
class Rotate(SpatialImageAugmentation):
    def forward_batch_img(self, tensor_image):
        rotate_degrees = self.rotate_radians(tensor_image.size(0)).view(-1) * 180/math.pi
        return kornia.geometry.rotate(tensor_image, rotate_degrees)


@aug_distributions(scales=Uniform2D(location=(.5, 1.5)))
class Scale(SpatialImageAugmentation):
    """Implementation of augmentation by scaling images.
    """
    def forward_sample_img(self, tensor_image):
        scale_x, scale_y = self.scales()
        result = F.interpolate(tensor_image.unsqueeze(dim=0), scale_factor=(scale_x, scale_y), mode='bilinear', align_corners=True)
        return result[0, :, :, :]


@aug_distributions(horizontal=Bernoulli(.5), vertical=Bernoulli(.5))
class Flip(SpatialImageAugmentation):
    def forward_sample_img(self, tensor_image):
        flip_dims = [False, self.horizontal(), self.vertical()]
        flip_dims = [n for n in range(len(flip_dims)) if flip_dims[n]]
        return tensor_image.flip(dims=flip_dims)


@aug_distributions(rectangle_size=Uniform2D((0.0, 1.0, 0.0, 1.0)), rectangle_center=Uniform2D((0.0, 1.0)))
class EraseRectangle(SpatialImageAugmentation):
    def forward_sample_img(self, tensor_image):
        _, _, img_width, img_height = tensor_image.size()
        result = tensor_image.clone()
        rect_width, rect_height = self.rectangle_size.get_rect_sizes(image_total_size=(img_width, img_height))
        rect_left, rect_top = self.rectangle_size.get_rect_locations(rect_sizes=(rect_width, rect_height),
                                                                     image_total_size=(img_width, img_height))
        result[:, rect_left:rect_left + rect_width, rect_left:rect_left + rect_width] = 0
        return result


@aug_distributions(horiz_center=Uniform2D((0.0, 1.0)), image_size=(224, 224))
class CropToSize(SpatialImageAugmentation):
    def forward_sample_img(self, tensor_image):
        _, _, img_width, img_height = tensor_image.size()

        result = tensor_image.clone()
        rect_width, rect_height = self.rectangle_size.get_rect_sizes(image_total_size=(img_width, img_height))
        rect_left, rect_top = self.rectangle_size.get_rect_locations(rect_sizes=(rect_width, rect_height),
                                                                     image_total_size=(img_width, img_height))
        result[:, rect_left:rect_left + rect_width, rect_left:rect_left + rect_width] = 0
        return result


#@aug_parameters(desired_width=224, desired_height=224)
@aug_distributions(pad_center=Uniform2D((0.0, 1.0)), crop_center=Uniform2D((0.0, 1.0)), image_size=(224, 224))
class CropPadAsNeeded(SpatialImageAugmentation):
    def forward_sample_img(self, tensor_image):
        _, img_width, img_height = tensor_image.size()
        tensor_image = tensor_image.unsqueeze(dim=0)
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

        return new_tensor_image[0, :, :, :]


#@aug_parameters(desired_width=224, desired_height=224, preserve_aspect_ratio=True)
# class ScaleAndPadAsNeeded(SpatialImageAugmentation):
#     def forward_sample_img(self, tensor_image):
#         tensor_image = tensor_image.unsqueeze(dim =0)
#         _, nb_channels, input_width, input_height = tensor_image.size()
#         if self.preserve_aspect_ratio:
#             new_tensor_image = torch.zeros(1, nb_channels, self.desired_width, self.desired_height)
#             if input_width/input_height > self.desired_width/self.desired_height: # scaling to desired width
#                 scaled_heigth = int(round(self.desired_height * (input_height/ input_width)))
#                 scaled_tensor_image = new_tensor_image = F.interpolate(tensor_image,
#                                                                        size=[self.desired_width, scaled_heigth],
#                                                                        mode='bilinear', align_corners=True)
#                 top = torch.randint(0, scaled_heigth - self.desired_height, (1,)).item()
#                 bottom = self.desired_height - top
#                 new_tensor_image[:,:,:,top:bottom] = scaled_tensor_image
#             else: #scaling to desired width
#                 scaled_width = int(round(self.desired_width * (input_width/input_height)))
#                 scaled_tensor_image = new_tensor_image = F.interpolate(tensor_image,
#                                                                        size=[scaled_width, self.desired_height],
#                                                                        mode='bilinear', align_corners=True)
#                 print(scaled_width, self.desired_width)
#                 if scaled_width == self.desired_width:
#                     left = 0
#                 else:
#                     left = torch.randint(0, scaled_width - self.desired_width, (1,)).item()
#                 right = self.desired_width - left
#                 new_tensor_image[:,:, left:right, :] = scaled_tensor_image
#         else:
#             new_tensor_image = F.interpolate(tensor_image, size=[self.desired_width, self.desired_width],
#                                              mode='bilinear', align_corners=True)
#         return new_tensor_image[0, :, :, :]
