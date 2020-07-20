import itertools
import numbers
import time
import types
from typing import Type
import torch
from matplotlib import pyplot as plt


from .base_augmentation import DeterministicImageAugmentation, SpatialImageAugmentation, StaticImageAugmentation


class AugmentedDs(torch.utils.data.Dataset):
    def __init__(self, ds, exemplar_augmentation: Type[DeterministicImageAugmentation], add_mask=False):
        self.ds = ds
        self._exemplar_augmentation = exemplar_augmentation
        self.add_mask = add_mask

    def new_augmentation(self):
        return self._exemplar_augmentation.like_me()

    def augment_sample(self, *args, augmentation=None):
        if augmentation is None:
            augmentation = self.new_augmentation()
        input_img = args[0]
        width = input_img.size(-2)
        height = input_img.size(-1)
        augmentated_data = []
        if self.add_mask:
            args.append(torch.ones_like(input_img[:1, :, :]).long())
        for datum in args:
            if isinstance(datum, torch.Tensor) and datum.size(-2) == width and datum.size(-1) == height:
                augmentated_data.append(augmentation(datum))
            else:
                augmentated_data.append(datum)
        return augmentated_data

    def __getitem__(self, item):
        return self.augment_sample(*self.ds[item])

    def __len__(self):
        return len(self.ds)


class AugmentedCocoDs(AugmentedDs):
    def __init__(self, ds, exemplar_augmentation: Type[DeterministicImageAugmentation], device="cpu", add_mask=False):
        super().__init__(ds=ds, exemplar_augmentation=exemplar_augmentation, add_mask=add_mask)
        self.device = device

    @staticmethod
    def rle2mask(rle_counts, size):
        mask_vector = torch.tensor(
            list(itertools.chain.from_iterable((((n) % 2,) * l for n, l in enumerate(rle_counts)))))
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

    def augment_sample(self, input, target, augmentation=None):
        if augmentation is None:
            augmentation = self.new_augmentation()
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
                obj_mask_images.append(object_mask.to(self.device))
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
        # if self.add_mask:
        #    mask_images.append(torch.ones([1, 1, input.size(-2), input.size(-1)], device=self.device))
        pc = (torch.tensor(point_cloud_x, device=self.device), torch.tensor(point_cloud_y, device=self.device))
        input = input.to(self.device)  # making a batch from an image
        aug_pc, aug_img = augmentation(pc, input)
        if len(obj_mask_images):
            obj_mask_images = torch.cat(obj_mask_images, dim=1).float()
            aug_obj_masks = augmentation(obj_mask_images)
        if self.add_mask:
            created_mask = torch.ones([1, input.size(-2), input.size(-1)], device=self.device)
            augmented_created_mask = augmentation(created_mask, is_mask=True)
        print("pc:", pc[0].size())
        print("input:",input.size())
        print("aug_img:", aug_img.size())

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
        if self.add_mask:
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
            ax3.imshow(mask[0, :, :].cpu(), cmap="gray")
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


def _get_augmentation_subjects_from_dataset(dataset, where, augmentation):
    """Guesses which images are image tensors.
    """
    sample = dataset[0]
    if where is "guess":
        if isinstance(augmentation, SpatialImageAugmentation):
            width, height = sample[0].size()[-2:]
            apply_on = []
            for t in sample:
                if isinstance(t, torch.Tensor) and len(t.size()) >= 2 and t.size(-2) == width and t.size(-1) == height:
                    apply_on.append(True)
                else:
                    apply_on.append(False)
            return tuple(apply_on)
        elif isinstance(augmentation, StaticImageAugmentation):
            # channel augmentations are only applicable on the input image
            return (True,) + (False,) * (len(dataset[0]) - 1)

    elif where == "all":
        return (True,) * len(dataset[0])
    elif where == "first" or where == 0:
        return (True,) + (False,) * (len(dataset[0]) - 1)
    elif where == "last" or where == -1:
        return (True,) + (False,) * (len(dataset[0]) - 1)
    elif isinstance(where, numbers.Integral):
        res = [False] * len(dataset[0])
        res[where] = True
        return tuple(res)
    elif isinstance(where, (list, tuple)):
        return tuple(where)
    else:
        raise ValueError()


class AugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentation_factory, apply_on="guess", append_input_map=False):
        self.dataset = dataset
        self.augmentation_factory = augmentation_factory
        self.apply_on = _get_augmentation_subjects_from_dataset(dataset, where=apply_on,
                                                                augmentation=augmentation_factory())
        self.append_input_map = append_input_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        augmentation = self.augmentation_factory.create_persistent()
        sample = self.dataset[item]
        sample = tuple([augmentation(sample[n]) if self.apply_on[n] else sample[n] for n in range(len(sample))])
        if self.append_input_map:
            sample = sample + (augmentation(torch.ones_like(sample[0])),)
        return sample


class ImageAugmentationPipelineDataset(torch.utils.data.Dataset):
    """Augments a pytorch dataset with some augmentations

        A dataset is assumed to be an oredered collection of 4D tensors (images), tensors of other dimensions and
        numbers float or integers. Images are assumed to be (Batch x Channels x Width x Height) and to always have the
        batch dimension with a size of 1. For segmentation data to be properly augmented, pixel labels must be encoded
        as one-hot encoding along the channel dimension.
    """

    def __init__(self, dataset, augmentations, apply_on="guess", occurrence_prob=1.0, append_input_map=False,
                 train=None):
        """Augments a pytorch dataset with some augmentations.

        :param dataset: A pytorch dataset to be wrapped and augmented.
        :param augmentations: One or more augmentation class or a factory generating instances of deterministic
            augmetions to be applied on each sample.
        :param apply_on: One of  ["all", "first", "last", "guess"] or a tuple or list of booleans. It describes on
            witch elements of a sample to apply the augmentation on. Guessing is done on the the assumption that the
            first item in a sample is an image (4D tensor) and the last two dimensions are width and height. All tensors
            having the last two dimensions with this size are assumed to be tensors to be augmented.
        :param occurrence_prob: The probability of augmenting the sample.
        :param append_input_map: If True, the sample will be appended by an image of ones wich will then be passed in
            the augmentation; after augmentation these images endout what is the contribution of the original image to
            each pixel after the augmentation.
        """
        if train is None:
            try:
                self.train = dataset.train
            except AttributeError:
                self.train = None
        else:
            self.train = train
        self.apply_on = _get_augmentation_subjects_from_dataset(dataset, where=apply_on,
                                                                augmentation=augmentations[0]())
        self.append_input_map = append_input_map
        if append_input_map:
            apply_on.append(True)
        self.apply_on = apply_on
        self.dataset = dataset
        self.occurrence_prob = occurrence_prob
        if not isinstance(augmentations, (list, tuple)):
            augmentations = (augmentations,)
        self.augmentation_factories = []

        for augmentation in augmentations:
            if isinstance(augmentation, types.LambdaType):
                self.augmentation_factories.append(augmentation)
            elif issubclass(augmentation, SpatialImageAugmentation):
                self.augmentation_factories.append(augmentation.factory())
            else:
                raise ValueError(
                    "augmentation must be either a DeterministicImageAugmentation or a lambda producing them.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = list(self.dataset[item])
        if self.append_input_map:
            sample.append(torch.ones_like(sample[0]))
        if torch.rand(1).item() > self.occurrence_prob:
            return sample
        else:
            augmentations = [f() for f in self.augmentation_factories]
            augmented_sample = []
            for should_apply, sample_tensor in zip(self.apply_on, sample):
                if should_apply:
                    for augmentation in augmentations:
                        sample_tensor = augmentation(sample_tensor)
                    augmented_sample.append(sample_tensor)
                else:
                    augmented_sample.append(sample_tensor)
            return augmented_sample
