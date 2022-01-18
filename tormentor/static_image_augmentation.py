import torch
from .sampling_fileds import SamplingField
from .deterministic_image_augmentation import DeterministicImageAugmentation, PointCloud, PointCloudList, PointCloudsImages, AugmentationState
from typing import Tuple


class StaticImageAugmentation(DeterministicImageAugmentation):
    r"""Parent class for augmentations that don't move things around.

    All classes that do descend from this are expected to be neutral for pointclouds, sampling fields although they
    might be erasing regions of the images so they are not gurantied to be neutral to masks.
    Every class were image pixels move around rather that just stay static, should be a descendant of
    SpatialImageAugmentation.
    """

    @classmethod
    def functional_image(cls, image_tensor, *state):
        raise NotImplementedError()

    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        return coords

    def forward_bboxes(self, bboxes: torch.FloatTensor, image_tensor=None, width_height=None):
        return bboxes

    def forward_pointcloud(self, pc: PointCloud, batch_tensor: torch.FloatTensor, compute_img: bool) -> PointCloud:
        if compute_img:
            return pc, self.forward_img(batch_tensor)
        else:
            return pc, batch_tensor

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        state = self.generate_batch_state(batch_tensor)
        return type(self).functional_image(*((batch_tensor,) + state))

    def forward_img_counterfactuals(self, batch_tensor: torch.FloatTensor, probs=torch.FloatTensor, nb_samples:int=-1) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if nb_samples == -1 or nb_samples >= probs.size(0):
            return self.forward_img(batch_tensor), probs
        else:
            idx = torch.argsort(probs * torch.rand(probs.size(0)))[-nb_samples:]
            return self.forward_img(batch_tensor[idx, :, :, :]), probs[idx]

    def forward_mask_counterfactuals(self, batch_tensor: torch.FloatTensor, probs=torch.FloatTensor, nb_samples:int=-1) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if nb_samples == -1 or nb_samples >= probs.size(0):
            return self.forward_mask(batch_tensor), probs
        else:
            idx = torch.argsort(probs * torch.rand(probs.size(0)))[-nb_samples:]
            return self.forward_mask(batch_tensor[idx, :, :, :]), probs[idx]

    def forward_bbox_counterfactuals(self, batch_tensor: torch.FloatTensor, image_tensor=None, width_height=None, probs=torch.FloatTensor, nb_samples:int=-1) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if nb_samples == -1 or nb_samples >= probs.size(0):
            return self.forward_bbox(batch_tensor, image_tensor, width_height), probs
        else:
            raise NotImplemented  #  bboxes across multiple images might be multiplexed

    def forward_pointcloud_counterfactuals(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor,
                                           compute_img: bool) -> Tuple[PointCloudsImages, torch.FloatTensor]:
        raise NotImplementedError()

    def forward_sampling_field_counterfactuals(self, coords: SamplingField, probs=torch.FloatTensor, nb_samples:int=-1) -> Tuple[SamplingField, torch.FloatTensor]:
        raise NotImplementedError()

class Identity(StaticImageAugmentation):
    def generate_batch_state(self, batch_tensor: torch.Tensor) -> AugmentationState:
        return ()

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        return batch_tensor


