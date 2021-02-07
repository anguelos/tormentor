import itertools
import time
from typing import Type
import torch
from matplotlib import pyplot as plt
from .base_augmentation import DeterministicImageAugmentation
from .factory import AugmentationFactory


class AugmentedDs(torch.utils.data.Dataset):
    r"""Wraps a Dataset in order to create an augmented dataset.

    .. code-block :: python

        import torchvision
        import tormentor
        import torch

        def augment_sample(sample, aug, device):
            print("Augmenting:",len(sample))
            return (aug(sample[0]).to(device),) + sample[1:]

        transform = torchvision.transforms.ToTensor()
        ds = torchvision.datasets.CIFAR10(download=True,train=False,root="/tmp", transform=transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=5)

        aug_ds = tormentor.AugmentedDs(ds, tormentor.RandomRotate, computation_device="cpu",augment_sample_function=augment_sample)
        aug_dl = torch.utils.data.DataLoader(aug_ds, batch_size=5)

        batch = next(iter(dl))
        aug_batch = next(iter(aug_dl))

        fig, ax = plt.subplots(2,1)
        ax[0].imshow(torchvision.utils.make_grid(batch[0]).numpy().transpose(1,2,0))
        ax[1].imshow(torchvision.utils.make_grid(aug_batch[0]).numpy().transpose(1,2,0))
        plt.show()
    """

    @staticmethod
    def augment_sample(args, augmentation:DeterministicImageAugmentation, device: torch.device):
        input_img = args[0]
        width = input_img.size(-2)
        height = input_img.size(-1)
        augmentated_data = []
        for n, datum in enumerate(args):
            if isinstance(datum, torch.Tensor) and datum.size(-2) == width and datum.size(-1) == height:
                augmentated_data.append(augmentation(datum, is_mask=n > 0))
            else:
                augmentated_data.append(datum)
        return augmentated_data

    def __init__(self, ds, augmentation_factory:AugmentationFactory, add_mask=False, computation_device="cpu", output_device=None, augment_sample_function=None):
        self.ds = ds
        self.augmentation_factory = augmentation_factory
        self.add_mask = add_mask
        self.computation_device = computation_device
        self.output_device = output_device
        if augment_sample_function is None:
            self.augment_sample_function = AugmentedDs.augment_sample
        else:
            self.augment_sample_function = augment_sample_function

    def new_augmentation(self):
        return self.augmentation_factory()

    def __getitem__(self, item):
        sample = self.ds[item]
        if self.add_mask:
            created_mask = torch.ones([1, sample[0].size(-2), sample[0].size(-1)], device=self.computation_device)
            sample = tuple(sample) + (created_mask,)
        augmentation = self.new_augmentation()
        result = self.augment_sample_function(sample, augmentation, device=self.computation_device)
        if self.output_device is not None and self.output_device != self.computation_device:
            return [data.to(self.output_device) if isinstance(data, torch.Tensor) else data for data in result]
        else:
            return result

    def __len__(self):
        return len(self.ds)


class AugmentedCocoDs(AugmentedDs):
    def __init__(self, ds, augmentation_factory: AugmentationFactory, computation_device="cpu",  output_device="cpu", add_mask=False):
        super().__init__(ds=ds, augmentation_factory=augmentation_factory, add_mask=add_mask, computation_device=computation_device, output_device=output_device)

    @staticmethod
    def rle2mask(rle_counts, size):
        mask_vector = torch.tensor(
            list(itertools.chain.from_iterable(((n % 2,) * l for n, l in enumerate(rle_counts)))))
        mask = mask_vector.view(size[1], size[0])
        mask = mask.transpose(0, 1).unsqueeze(dim=0).unsqueeze(dim=0)
        return mask

    @staticmethod
    def mask2rle(mask):
        mask_vector = (mask[0, 0, :, :] > 0.5).transpose(0, 1).reshape(-1)
        change = torch.empty_like(mask_vector)
        change[0] = False
        change[1:] = mask_vector[1:] != mask_vector[:-1]
        dense_change = torch.arange(change.size(0))[change]
        rle = torch.empty_like(dense_change)
        rle[0] = dense_change[0]
        rle[1:] = dense_change[1:] - dense_change[:-1]
        rle = rle.tolist()
        rle.append(mask_vector.size(0) - dense_change[-1])
        size = [mask.size(2), mask.size(3)]
        return rle, size

    @classmethod
    def augment_sample(cls, input, target, mask, augmentation, device):
        point_cloud_x = []
        point_cloud_y = []
        n = 0
        object_start_end_pos = []
        obj_mask_images = []
        for object_n, coco_object in enumerate(target):
            if coco_object["iscrowd"]:
                object_start_end_pos.append(None)
                object_mask = AugmentedCocoDs.rle2mask(coco_object["segmentation"]["counts"],
                                                coco_object["segmentation"]["size"])
                obj_mask_images.append(object_mask.to(device))
            else:
                object_surfaces = []
                for surface_n in range(len(coco_object["segmentation"])):
                    start_pos = n
                    X = torch.Tensor(coco_object["segmentation"][surface_n])[::2]
                    Y = torch.Tensor(coco_object["segmentation"][surface_n])[1::2]
                    point_cloud_x += X.tolist()
                    point_cloud_y += Y.tolist()
                    n += X.size(0)
                    end_pos = n
                    object_surfaces.append((start_pos, end_pos))
                object_start_end_pos.append(object_surfaces)
        pc = (torch.tensor(point_cloud_x, device=device), torch.tensor(point_cloud_y, device=device))
        input = input.to(device)  # making a batch from an image
        aug_pc, aug_img = augmentation(pc, input)
        if len(obj_mask_images):
            obj_mask_images = torch.cat(obj_mask_images, dim=1).float()
            aug_obj_masks = augmentation(obj_mask_images)
        if mask is not None:
            #created_mask = torch.ones([1, input.size(-2), input.size(-1)], device=self.device)
            augmented_created_mask = augmentation(mask, is_mask=True)

        X, Y = aug_pc
        aug_target = []
        current_mask = 0
        for object_n, surface_start_end_pos in enumerate(object_start_end_pos):
            if target[object_n]["iscrowd"]:
                # TODO (anguelos) is there a more elegant way to get left, top, width, and height?
                obj_mask = aug_obj_masks[:, current_mask: current_mask + 1, :, :]
                area = obj_mask.size()
                y = obj_mask[0, 0, :, :].sum(dim=0) > 0
                x = obj_mask[0, 0, :, :].sum(dim=1) > 1
                y = torch.where(y)[0]
                x = torch.where(x)[0]
                left = x.min()
                right = x.max()
                width = right - left
                top = y.min()
                bottom = y.max()
                height = bottom - top
                rle, size = AugmentedCocoDs.mask2rle(obj_mask)
                segmentation = {"counts": rle, "size": size}
                current_mask += 1
            else:
                object_range = object_start_end_pos[object_n][0][0], object_start_end_pos[object_n][-1][1]
                left = X[object_range[0]: object_range[1]].min().item()
                right = X[object_range[0]: object_range[1]].max().item()
                width = right - left
                top = Y[object_range[0]: object_range[1]].min().item()
                bottom = Y[object_range[0]: object_range[1]].max().item()
                height = bottom - top
                segmentation = []
                for surface_n, (surface_start, surface_end) in enumerate(surface_start_end_pos):
                    surf_X, surf_Y = X[surface_start:surface_end], Y[surface_start:surface_end]
                    segmentation.append(torch.cat([surf_X.view(-1, 1), surf_Y.view(-1, 1)], dim=1).view(-1).tolist())
                area = target[object_n]["area"]  # TODO (anguelos) recompute area with a polygon library
            object = {"area": area,
                      "iscrowd": target[object_n]["iscrowd"],
                      "image_id": target[object_n]["image_id"],
                      "category_id": target[object_n]["category_id"],
                      "id": target[object_n]["id"],
                      "segmentation": segmentation,
                      "bbox": [left, top, width, height]}
            aug_target.append(object)
        if mask is not None:
            return aug_img, aug_target, augmented_created_mask
        else:
            return aug_img, aug_target

    def show_augmentation(self, n, augmentation=None, save_filename=None):
        if augmentation is None:
            augmentation = self.new_augmentation()
        simple_input, simple_target = self.ds[n]
        t = time.time()
        if self.add_mask:
            augmented_input, augmented_target, mask = self.augment_sample(simple_input, simple_target,
                                                                          augmentation=augmentation)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax3.imshow(mask[0, :, :].cpu(), cmap="gray",vmin=.0, vmax=1.)
            ax3.set_title(f"Mask")
            ax3.set_xticks([])
            ax3.set_xticks([], minor=True)
            ax3.set_yticks([])
            ax3.set_yticks([], minor=True)
        else:
            augmented_input, augmented_target = self.augment_sample(simple_input, simple_target,
                                                                    augmentation=augmentation)
            fig, (ax1, ax2) = plt.subplots(1, 2)
        duration = time.time() - t

        ax1.imshow(simple_input.cpu().transpose(2, 0).transpose(1, 0))
        ax1.set_title(f"Input ({n})")
        ax1.set_xticks([])
        ax1.set_xticks([], minor=True)
        ax1.set_yticks([])
        ax1.set_yticks([], minor=True)
        ax2.imshow(augmented_input.cpu().transpose(2, 0).transpose(1, 0))
        ax2.set_title(f"Augmentation: {repr(augmentation)}.\nComputed in {duration:.4f} sec.")
        ax2.set_xticks([])
        ax2.set_xticks([], minor=True)
        ax2.set_yticks([])
        ax2.set_yticks([], minor=True)
        for coco_object in simple_target:
            if not coco_object["iscrowd"]:
                x = torch.Tensor(coco_object["segmentation"][0][::2] + [coco_object["segmentation"][0][0]])
                y = torch.Tensor(coco_object["segmentation"][0][1::2] + [coco_object["segmentation"][0][1]])
                ax1.plot(x, y)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.set_ticks_position('none')
                l, t, w, h = coco_object["bbox"]
                r, b = l + w, t + h
        for coco_object in augmented_target:
            if not coco_object["iscrowd"]:
                x = torch.Tensor(coco_object["segmentation"][0][::2] + [coco_object["segmentation"][0][0]])
                y = torch.Tensor(coco_object["segmentation"][0][1::2] + [coco_object["segmentation"][0][1]])
                ax2.plot(x, y)
                ax2.xaxis.set_ticks_position('none')
                ax2.yaxis.set_ticks_position('none')
                l, t, w, h = coco_object["bbox"]
                r, b = l + w, t + h
        if save_filename is None:
            plt.show()
        else:
            plt.savefig(save_filename)


def labels2onehots(label_sample_tensor, n_classes:int, neutral_label:int):
    if neutral_label >1:
        label_sample_tensor = label_sample_tensor.copy().unsqueeze(dim=0)
        height, width = label_sample_tensor
        onehot = torch.zeros(n_classes + 2, height, width)
        neutral_slice = label_sample_tensor == neutral_label
        label_sample_tensor[neutral_slice] = 0
        onehot.scatter_(dim=0,  index=label_sample_tensor.unsqueeze(dim=0), src=torch.ones(label_sample_tensor.size()).unsqueeze(dim=0))
        return onehot
    else:
        label_sample_tensor = label_sample_tensor.copy().unsqueeze(dim=0)
        #onehot.scatter_(dim=0, index=label_sample_tensor.unsqueeze(dim=0), src=torch.ones(label_sample_tensor.size()).unsqueeze(dim=0))
        height, width = label_sample_tensor
        # based on https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
        #y = torch.LongTensor(batch_size, 1).random_() % nb_digits
        onehot = torch.zeros(n_classes + 2, height, width)
        neutral_slice = label_sample_tensor == neutral_label
        label_sample_tensor[neutral_slice] = 0
        onehot.scatter_(dim=0,  index=label_sample_tensor.unsqueeze(dim=0), src=torch.ones(label_sample_tensor.size()).unsqueeze(dim=0))
        return onehot



class AugmentedVOCSegmentationDs(AugmentedDs):
    def __init__(self, ds, exemplar_augmentation: Type[DeterministicImageAugmentation], device="cpu", add_mask=False,
                 n_classes=20, neutral_label=255, zero_bias=.001):
        super().__init__(ds=ds, exemplar_augmentation=exemplar_augmentation, add_mask=add_mask)
        self.device = device

    def __getitem__(self, item):
        input, segmentation = self.ds[item]
        segmentation = (segmentation * 255).long()
