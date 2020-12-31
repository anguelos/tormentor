from typing import Tuple, Union, List, Type
from .base_augmentation import DeterministicImageAugmentation
from .factory import AugmentationFactory
import torch

Tensors = Union[List[torch.Tensor], Tuple[torch.Tensor]]

class AugmentedDataLoader(torch.utils.data.DataLoader):
    r"""Wraps a dataloader in order to create an augmented dataloader.

    .. code-block :: python

        import torchvision
        import tormentor
        import torch

        def augment_batch(batch,aug,device):
            return [aug(batch[0]).to(device)]+batch[1:]

        transform = torchvision.transforms.ToTensor()
        ds = torchvision.datasets.CIFAR10(download=True,train=False,root="/tmp", transform=transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=5)
        aug_dl = tormentor.AugmentedDataLoader(dl, tormentor.RandomRotate, computation_device="cpu",
        augment_batch_function=classification_augment)

        batch = next(iter(dl))
        aug_batch = next(iter(aug_dl))

        fig, ax = plt.subplots(2,1)
        ax[0].imshow(torchvision.utils.make_grid(batch[0]).numpy().transpose(1,2,0))
        ax[1].imshow(torchvision.utils.make_grid(aug_batch[0]).numpy().transpose(1,2,0))
        plt.show()
    """

    @staticmethod
    def augment_batch(batch: Tensors, augmentation: DeterministicImageAugmentation, process_device: torch.device):
        input_imgs, segmentations, masks = batch
        if process_device != batch[0].device:
            input_imgs = input_imgs.to(process_device)
            segmentations = segmentations.to(process_device)
            masks = masks.to(process_device)
        with torch.no_grad():
            input_imgs = augmentation(input_imgs)
            segmentations = augmentation(segmentations, is_mask=True)
            masks = augmentation(masks, is_mask=True)
            segmentations = torch.clamp(segmentations[:, :1, :, :] + (1 - masks), 0., 1.0)
            segmentations = torch.cat([segmentations, 1 - segmentations], dim=1)
            if process_device != batch[0].device:
                return input_imgs.to(batch[0].device), segmentations.to(batch[0].device), masks.to(batch[0].device)
            else:
                return input_imgs, segmentations, masks

    def __init__(self, dl: torch.utils.data.DataLoader, augmentation_factory: AugmentationFactory, computation_device: torch.device,
                 augment_batch_function=None, output_device=None):
        self.dl = dl
        self.augmentation_factory = augmentation_factory
        self.computation_device = computation_device
        self.output_device = output_device
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
        augmentation = self.augmentation_factory()
        output_batch = self.augment_batch_function(next(self.iterator), augmentation, self.computation_device)
        if self.output_device is not None and self.output_device != self.computation_device:
            output_batch = output_batch.to(self.output_device)
        return output_batch
