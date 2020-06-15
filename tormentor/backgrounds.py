import torch
import PIL
import kornia as K
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


def load_images_as_tensors(image_filename_list, channel_count, min_width_height, preserve_aspect_ratio=True):
    assert channel_count in (1, 2, 3, 4)
    image_list = []
    for image_file_name in image_filename_list:
        img = PIL.Image.open(image_file_name)
        if channel_count == 1:
            img = img.convert("L")
        elif channel_count == 2:
            img = img.convert("LA")
        elif channel_count == 3:
            img = img.convert("RGB")
        else:  # channel_count == 4
            img = img.convert("RGBA")
        width, height = img.size
        if min_width_height is not None and (width < min_width_height[0] or height < min_width_height[1]):
            if preserve_aspect_ratio:
                width, height = img
                width_scale = min_width_height[0] / width
                height_scale = min_width_height[1] / height
                scale = max(width_scale, height_scale)
                img = img.resize((int(scale * width), int(scale * height)))
            else:
                img = img.resize(min_width_height)
        image_list.append(K.image_to_tensor(img).unsqueeze(dim=0))
    return image_filename_list

class BackgroundChoice(AbstractBackground):
    image_id = Categorical(1000)

    @classmethod
    def create(cls, image_filename_list, channel_count, min_width_height, requires_grad=False):




        all_distributions = {"choice": Categorical(len(augmentation_list), requires_grad=requires_grad)}
        for augmentation in augmentation_list:
            class_name = str(augmentation).split(".")[-1][:-2]
            cls_distributions = augmentation.get_distributions()
            cls_distributions = {f"{class_name}_{k}": v for k, v in cls_distributions.get_items()}
            all_distributions.update(cls_distributions)
        for cls_distribution in cls_distributions:
            for parameter in cls_distribution.get_distribution_parameters():
                parameter.requires_grad_(requires_grad)
        new_cls_name = f"{cls.__qualname__}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), cls_distributions)
        return new_cls


    def forward_batch_img(self, tensor_image):
        batch_size = tensor_image.size(0)
        image_id = type(self).image_id(batch_size)
