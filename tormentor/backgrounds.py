from .util import load_images_as_tensors
import torch
from diamond_square import functional_diamond_square
from .random import Uniform, Bernoulli, Categorical, Constant, Normal
from .base_augmentation import DeterministicImageAugmentation


class AbstractBackground(DeterministicImageAugmentation):
    def blend_by_mask(self, input_tensor, mask_tensor):
        res = input_tensor * mask_tensor + (1 - mask_tensor) * self.like(input_tensor).to(input_tensor.device)
        return res

class ConstantBackground(AbstractBackground):
    value = Constant(0)

    def forward_batch_img(self, tensor_image):
        return torch.zeros_like(tensor_image) + type(self).value().to(tensor_image.device)


class UniformNoiseBackground(AbstractBackground):
    pixel_values = Uniform(value_range=(0.0, 1.0))

    def forward_batch_img(self, tensor_image):
        return type(self).pixel_values(tensor_image.size(), device=tensor_image.device)


class NormalNoiseBackground(AbstractBackground):
    pixel_values = Normal(mean=0.0, deviation=1.0)

    def forward_batch_img(self, tensor_image):
        return type(self).pixel_values(tensor_image.size(), device=tensor_image.device)


class PlasmaBackground(AbstractBackground):
    roughness = Uniform(value_range=(0.2, 0.6))
    pixel_means = Uniform(value_range=(0.0, 1.0))
    pixel_ranges = Uniform(value_range=(0.0, 1.0))

    def forward_batch_img(self, tensor_image):
        batch_size = tensor_image.size(0)
        roughness = type(self).roughness(batch_size)
        pixel_means = type(self).pixel_means(batch_size, device=tensor_image.device).view(-1, 1, 1, 1)
        pixel_ranges = type(self).pixel_ranges(batch_size, device=tensor_image.device).view(-1, 1, 1, 1)
        plasma = functional_diamond_square(tensor_image.size(), roughness=roughness, device=tensor_image.device)
        plasma_reshaped = plasma.reshape([batch_size, -1])
        plasma_min = plasma_reshaped.min(dim=1)[0].view(-1, 1, 1, 1)
        plasma_max = plasma_reshaped.max(dim=1)[0].view(-1, 1, 1, 1)
        return (pixel_ranges * (plasma - plasma_min) / (plasma_max - plasma_min)) + pixel_means - pixel_ranges / 2


class BackgroundImageDB(AbstractBackground):
    image_id = Categorical(1)
    center_x = Uniform((0., 1.))
    center_y = Uniform((0., 1.))

    @classmethod
    def create(cls, image_filename_list, channel_count, min_width_height, preserve_aspect_ratio, requires_grad=False):
        images=list(load_images_as_tensors(image_filename_list=image_filename_list, channel_count=channel_count, min_width_height=min_width_height, preserve_aspect_ratio=preserve_aspect_ratio))

        cls_distributions = {"image_id": Categorical(len(image_filename_list), requires_grad=requires_grad)}
        for cls_distribution in cls_distributions:
            for parameter in cls_distribution.get_distribution_parameters():
                parameter.requires_grad_(requires_grad)
        all_cls_members={"images":images}
        all_cls_members.update(cls_distributions)
        new_cls_name = f"{cls.__qualname__}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), all_cls_members)
        return new_cls

    def forward_batch_img(self, tensor_image):
        batch_size, channels, width, height = tensor_image.size(0)
        image_ids = type(self).image_id(batch_size)
        res = torch.empty(tensor_image.size())
        for n in range(batch_size):
            img = type(self).images[image_ids[n]]
            assert width <= img.size(2) and height <= img.size(3) and img.size(0)==channels
            x_offset = int((img.size(2)-width))
            y_offset = int((img.size(3) - height))
            res[n, :, :, :] = img[:,x_offset:x_offset+width, y_offset+y_offset:height]
        return res
