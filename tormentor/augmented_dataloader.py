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
        #input_imgs, segmentations, masks = batch
        batch = [tensor.to(process_device) for tensor in batch]
        input_imgs = batch[0]
        img_size = input_imgs.size()
        mask_size = (img_size[0], 1, img_size[2], img_size[3])
        batch_sz, nb_channels, width, height = input_imgs.size()
        augmented_batch = []
        with torch.no_grad():
            for datum in batch:
                if isinstance(datum, torch.Tensor) and datum.size() == img_size:
                    augmented_batch.append(augmentation(datum))
                elif isinstance(datum, torch.Tensor) and datum.size() == mask_size:
                    augmented_batch.append(augmentation(datum, is_mask=True))
                else:
                    augmented_batch.append(datum)
        return augmented_batch

    def __init__(self, dl: torch.utils.data.DataLoader, augmentation_factory: AugmentationFactory, computation_device: torch.device,
                 augment_batch_function=None, output_device=None, add_mask=False):
        self.dl = dl
        self.augmentation_factory = augmentation_factory
        self.computation_device = computation_device
        self.output_device = output_device
        self.add_mask = add_mask
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
        if self.add_mask:
            mask = torch.ones([output_batch.size(0), 1, output_batch.size(2), output_batch.size(3)])
            mask = augmentation(mask)
            output_batch = output_batch + (mask,)
        return output_batch
