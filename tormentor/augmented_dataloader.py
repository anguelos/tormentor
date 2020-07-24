from typing import Tuple, Union, List, Type


#SamplingField = Tuple[torch.FloatTensor, torch.FloatTensor]
#PointCloud = Tuple[torch.FloatTensor, torch.FloatTensor]
#PointCloudList = List[PointCloud]
#PointCloudsImages = Tuple[PointCloudList, torch.FloatTensor]


from .base_augmentation import DeterministicImageAugmentation

import torch

class AugmentedDataLoader(torch.utils.data.DataLoader):
    """

    """
    @staticmethod
    def augment_batch(batch, augmentation: DeterministicImageAugmentation, process_device: torch.device):
        input_imgs, segmentations, masks = batch
        if process_device != batch[0].device:
            input_imgs=input_imgs.to(process_device)
            segmentations=segmentations.to(process_device)
            masks=masks.to(process_device)
        with torch.no_grad():
            input_imgs = augmentation(input_imgs)
            segmentations = augmentation(segmentations, is_mask=True)
            masks = augmentation(masks, is_mask=True)
            segmentations = torch.clamp(segmentations[:,:1, :, :] + (1-masks),0.,1.0)
            segmentations = torch.cat([segmentations, 1-segmentations], dim=1)
            if process_device != batch[0].device:
                return input_imgs.to(batch[0].device), segmentations.to(batch[0].device), masks.to(batch[0].device)
            else:
                return input_imgs, segmentations, masks

    def __init__(self, dl: torch.utils.data.DataLoader, augmentation_cls: type, device: torch.device, augment_batch_function=None):
        self.dl = dl
        self.augmentation_cls = augmentation_cls
        self.device = device
        if augment_batch_function is None:
            self.augment_batch_function = AugmentedDataLoader.augment_batch
        else:
            self.augment_batch_function = augment_batch_function

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        self.iterator = iter(self.dl)
        return self

    def __next__(self):
        return self.augment_batch_function(next(self.iterator), self.augmentation_cls(), self.device)
