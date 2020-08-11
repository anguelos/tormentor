import kornia as K
import torch

from .base_augmentation import SamplingField, SpatialAugmentationState, DeterministicImageAugmentation
from .random import Constant, Uniform


class ResizingAugmentation(DeterministicImageAugmentation):
    out_width = Constant(224)
    out_height = Constant(224)

    @classmethod
    def new_size(cls, width: int, height: int, requires_grad=False):
        return cls.override_distributions(requires_grad=requires_grad, out_width=Constant(width),
                                          out_height=Constant(height))

    @staticmethod
    def resample(old_coords: SamplingField, new_coords: SamplingField):
        print("new_coords:",    new_coords[0].size())
        print("old_coords:", old_coords[0].size())
        new_coords_grid = torch.cat((new_coords[0].unsqueeze(dim=-1), new_coords[1].unsqueeze(dim=-1)), dim=3)
        collated_old_coords = torch.cat((old_coords[0].unsqueeze(dim=1), old_coords[1].unsqueeze(dim=1)), dim=1)
        collated_new_coords = torch.nn.functional.grid_sample(collated_old_coords, new_coords_grid, align_corners=True)
        print("collated_new_coords:", collated_new_coords.size())
        res = collated_new_coords[:, 0, :, :], collated_new_coords[:, 1, :, :]
        print("res:", res[0].size())
        return res

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> SpatialAugmentationState:
        raise NotImplementedError()

    @classmethod
    def functional_sampling_field(cls, coords: SamplingField, *state) -> SamplingField:
        raise NotImplementedError()

    @classmethod
    def functional_image(cls, img: torch.Tensor, *state) -> torch.Tensor:
        raise NotImplementedError()

    def forward_img(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        state = self.generate_batch_state(batch_tensor)
        return type(self).functional_image(*((batch_tensor,) + state))

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        state = self.generate_batch_state(coords[0].unsqueeze(dim=1))
        return type(self).functional_sampling_field(*((coords,) + state))

    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_img(X)


class PadTo(ResizingAugmentation):
    """Guaranties a minimum size by padding.

    """
    center_x = Uniform((0.0, 1.0))
    center_y = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> SpatialAugmentationState:
        batch_size, _, img_height, img_width = batch_tensor.size()
        out_width = type(self).out_width(1)
        out_height = type(self).out_height(1)
        center_x = type(self).center_x(batch_size)
        center_y = type(self).center_y(batch_size)
        if out_width > img_width:
            total_h_pad = out_width - img_width
            pad_left = torch.round((center_x * total_h_pad)).round().long().view(-1, 1)
            pad_right = total_h_pad - pad_left
        else:
            pad_left, pad_right = torch.zeros([batch_size, 1]), torch.zeros([batch_size, 1])

        if out_height > img_height:
            total_v_pad = out_height - img_height
            pad_top = torch.round((center_y * total_v_pad)).long().view(-1, 1)
            pad_bottom = total_v_pad - pad_top
        else:
            pad_top, pad_bottom = torch.zeros([batch_size, 1]), torch.zeros([batch_size, 1])

        ltrb_padings = torch.cat((pad_left, pad_top, pad_right, pad_bottom), dim=1).long()
        return ltrb_padings,

    @classmethod
    def functional_sampling_field(cls, coords: SamplingField, ltrb_padings: torch.Tensor) -> SamplingField:
        in_X, in_Y = coords
        in_X = in_X.unsqueeze(dim=1)
        in_Y = in_Y.unsqueeze(dim=1)
        batch_size, _, in_height, in_width = in_X.size()
        out_X = []
        out_Y = []
        for n in range(batch_size):
            left, top, right, bottom = ltrb_padings[n,:].tolist()
            cur_Y = torch.nn.functional.pad(in_Y[n:n+1,:, :, :], [left, 0, 0, 0], value=-1.1)
            cur_Y = torch.nn.functional.pad(cur_Y, [0, right, 0, 0], value=1.1)
            cur_Y = torch.nn.functional.pad(cur_Y, [0, 0, top, bottom], mode='replicate')
            out_Y.append(cur_Y)

            cur_X = torch.nn.functional.pad(in_X[n:n+1,:, :, :], [0, 0, top, 0], value=-1.1)
            cur_X = torch.nn.functional.pad(cur_X, [0, 0, 0, bottom], value=1.1)
            cur_X = torch.nn.functional.pad(cur_X, [left, right, 0, 0], mode='replicate')
            out_X.append(cur_X)
        out_X = torch.cat(out_X, dim=0)
        out_Y = torch.cat(out_Y, dim=0)
        return out_X[:, 0, :, :], out_Y[:,0,:,:]

    @classmethod
    def functional_image(cls, batch: torch.Tensor, ltrb_padings) -> torch.Tensor:
        res_tensors = []
        lrtb_paddings = ltrb_padings[:, [0, 2, 1, 3]]
        for n in range(batch.size(0)):
            res_tensors.append(torch.nn.functional.pad(batch[n:n + 1, :, :, :], lrtb_paddings[n, :].tolist()))
        res = torch.cat(res_tensors, dim=0)
        return res


class CropTo(ResizingAugmentation):
    center_x = Uniform((0.0, 1.0))
    center_y = Uniform((0.0, 1.0))

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> SpatialAugmentationState:
        batch_size, _, img_height, img_width = batch_tensor.size()
        out_width = type(self).out_width(1)
        out_height = type(self).out_height(1)
        center_x = type(self).center_x(batch_size)
        center_y = type(self).center_y(batch_size)
        if out_width < img_width:
            total_h_crop = img_width - out_width
            crop_left = torch.round((center_x * total_h_crop)).round().long().view(-1, 1)
            crop_right = total_h_crop - crop_left
        else:
            crop_left, crop_right = torch.zeros([batch_size, 1]), torch.zeros([batch_size, 1])

        if out_height < img_height:
            total_v_crop = img_height - out_height
            crop_top = torch.round((center_y * total_v_crop)).long().view(-1, 1)
            crop_bottom = total_v_crop - crop_top
        else:
            crop_top, crop_bottom = torch.zeros([batch_size, 1]), torch.zeros([batch_size, 1])

        ltrb_crops = torch.cat((crop_left, crop_top, crop_right, crop_bottom), dim=1).long()
        return ltrb_crops,

    @classmethod
    def functional_sampling_field(cls, coords: SamplingField, ltrb_crops: torch.Tensor) -> SamplingField:
        in_X, in_Y = coords
        batch_size, in_height, in_width = in_X.size()
        out_X = []
        out_Y = []
        for n in range(batch_size):
            left, top, right_crop, bottom_crop = ltrb_crops[n, :].tolist()
            right = in_width - right_crop
            bottom = in_height - bottom_crop
            out_Y.append(in_Y[n:n+1, top:bottom, left:right])
            out_X.append(in_X[n:n+1, top:bottom, left:right])
        out_X = torch.cat(out_X, dim=1)
        out_Y = torch.cat(out_Y, dim=1)
        return out_X, out_Y

    @classmethod
    def functional_image(cls, batch: torch.Tensor, ltrb_crops) -> torch.Tensor:
        res_tensors = []
        batch_size, _, in_height, in_width = batch.size()
        for n in range(batch.size(0)):
            left, top, right_crop, bottom_crop = ltrb_crops[n, :].tolist()
            right = in_width - right_crop
            bottom = in_height - bottom_crop
            res_tensors.append(batch[n: n + 1, :, top: bottom, left: right])
        res = torch.cat(res_tensors, dim=0)
        return res
