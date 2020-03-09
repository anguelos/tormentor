import torch
import glob
from diamond_square import diamond_square


class BackgroundGenerator(object):
    def create(self, size, device, dtype):
        raise NotImplementedError()

    def like(self, tensor):
        self.create(tensor.size(),device=tensor.device, dtype=tensor.dtype)


class ConstantBackground(BackgroundGenerator):
    def __init__(self, value=0):
        self.value = value

    def create(self, size, device, dtype):
        result = torch.zeros(size, device=device, dtype=dtype)+self.value


class UniformNoiseBackground(BackgroundGenerator):
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.value_range = max_value - min_value

    def create(self, size, device, dtype):
        return torch.rand(size,device=device,dtype=dtype)*self.value_range+self.min_value

class GaussianNoiseBackground(BackgroundGenerator):
    def __init__(self, mean=0.0, variance=1.0):
        self.mean=mean
        self.variance=variance

    def create(self, size, device, dtype):
        return torch.rand(size,device=device,dtype=dtype)*self.variance+self.mean


class ImageBackground(BackgroundGenerator):
    def __init__(self, image_filename_pattern):
        self.filenames = glob.glob(image_filename_pattern)

    def create(self, size, device, dtype):
        desired_batch_size, channels, desired_width, desired_height
        for sample_in_batch 
        new_tensor_image = torch.zeros(1, nb_channels, desired_width, desired_height)
        if input_width / input_height > self.desired_width / self.desired_height:  # scaling to desired width
            scaled_heigth = int(round(self.desired_height * (input_height / input_width)))
            scaled_tensor_image = new_tensor_image = F.interpolate(tensor_image,
                                                                   size=[self.desired_width, scaled_heigth],
                                                                   mode='bilinear', align_corners=True)
            top = torch.randint(0, scaled_heigth - self.desired_height, (1,)).item()
            bottom = self.desired_height - top
            new_tensor_image[:, :, :, top:bottom] = scaled_tensor_image
        else:  # scaling to desired width
            scaled_width = int(round(self.desired_width * (input_width / input_height)))
            scaled_tensor_image = new_tensor_image = F.interpolate(tensor_image,
                                                                   size=[scaled_width, self.desired_height],
                                                                   mode='bilinear', align_corners=True)
            print(scaled_width, self.desired_width)
            if scaled_width == self.desired_width:
                left = 0
            else:
                left = torch.randint(0, scaled_width - self.desired_width, (1,)).item()
            right = self.desired_width - left
            new_tensor_image[:, :, left:right, :] = scaled_tensor_image


class PlasmaBackground(BackgroundGenerator):
    def __init__(self, min_roughness=.3, max_roughness=.8, min_mean=0.0, max_mean=1.0, min_range=0.0, max_range=1.0):
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.min_range = min_range
        self.max_range = max_range

    def create(self, size, device, dtype):
        raise NotImplementedError()
