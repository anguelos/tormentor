from .random import Categorical
from .deterministic_image_augmentation import DeterministicImageAugmentation, SamplingField, PointCloudList, PointCloudsImages
import torch

class AugmentationChoice(DeterministicImageAugmentation):
    r"""Select randomly among many augmentations.

    .. figure :: _static/example_images/AugmentationChoice.png

        Random choice of perspective and plasma-brightness augmentations

        .. code-block :: python

            augmentation_factory = tormentor.RandomPerspective ^ tormentor.RandomPlasmaBrightness
            augmentation = augmentation_factory()
            augmented_image = augmentation(image)
    """

    @classmethod
    def create(cls, augmentation_list, requires_grad=False, new_cls_name=None):
        new_parameters = {"choice": Categorical(len(augmentation_list)), "available_augmentations": augmentation_list}
        for augmentation in augmentation_list:
            class_name = str(augmentation).split(".")[-1][:-2]
            cls_distributions = augmentation.get_distributions()
            cls_distributions = {f"{class_name}_{k}": v for k, v in cls_distributions.items()}
            new_parameters.update(cls_distributions)
        for cls_distribution in cls_distributions.values():
            for parameter in cls_distribution.get_distribution_parameters().values():
                parameter.requires_grad_(requires_grad)
        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        if new_cls_name is None:
            new_cls_name = f"{cls_oldname}_{torch.randint(1000000, 9000000, (1,)).item()}"
        new_cls = type(new_cls_name, (cls,), new_parameters)
        return new_cls

    def forward_sampling_field(self, coords: SamplingField):
        batch_sz = coords[0].size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch_x = []
        augmented_batch_y = []
        for sample_n in range(batch_sz):
            sample_coords = coords[0][sample_n: sample_n + 1, :, :], coords[1][sample_n: sample_n + 1, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            sample_x, sample_y = augmentation.forward_sampling_field(sample_coords)
            augmented_batch_x.append(sample_x)
            augmented_batch_y.append(sample_y)
        augmented_batch_x = torch.cat(augmented_batch_x, dim=0)
        augmented_batch_y = torch.cat(augmented_batch_y, dim=0)
        return augmented_batch_x, augmented_batch_y

    def forward_img(self, batch_tensor):
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            augmented_sample = augmentation.forward_img(sample_tensor)
            augmented_batch.append(augmented_sample)
        augmented_batch = torch.cat(augmented_batch, dim=0)
        return augmented_batch


    def forward_img_path_probabilities(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        probs = self.choice.probs[0, augmentation_ids]
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            probs[sample_n] *= augmentation.forward_img_path_probabilities(sample_tensor)[0]
        return probs


    def forward_mask(self, batch_tensor):
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            augmented_sample = augmentation.forward_mask(sample_tensor)
            augmented_batch.append(augmented_sample)
        augmented_batch = torch.cat(augmented_batch, dim=0)
        return augmented_batch

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor,
                           compute_img: bool) -> PointCloudsImages:
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        augmented_pcl = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            pc_onelist = pcl[sample_n: sample_n + 1]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            aug_pc_onelist, augmented_sample = augmentation.forward_pointcloud(pc_onelist, sample_tensor, compute_img)
            augmented_batch.append(augmented_sample)
            augmented_pcl = augmented_pcl + aug_pc_onelist
        if compute_img:
            augmented_batch = torch.cat(augmented_batch, dim=0)
            return augmented_pcl, augmented_batch
        else:
            return augmented_pcl, batch_tensor




