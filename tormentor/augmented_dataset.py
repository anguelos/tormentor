import types
import numbers
from matplotlib import pyplot as plt
import torch
import numpy as np

from .base_augmentation import DeterministicImageAugmentation, SpatialImageAugmentation, ChannelImageAugmentation
from typing import Type


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
        input = args[0]
        width = input.size(-2)
        height = input.size(-1)
        augmentated_data= []
        if self.add_mask:
            args.append(torch.ones_like(input[:1,:,:]))
        for datum in args:
            if isinstance(datum, torch.Tensor) and datum.size(-2) == width and datum.size(-1) == height:
                augmentated_data.append(augmentation(datum))
            else:
                augmentated_data.append(datum)
        return augmentated_data

    def __getitem__(self, item):
        return self.augment_sample(*self.ds[item])

    def __len__(self, item):
        return len(self.coco_ds)


class AugmentedCocoDs(AugmentedDs):
    def __init__(self, ds, exemplar_augmentation:Type[DeterministicImageAugmentation]):
        super().__init__(ds=ds, exemplar_augmentation=exemplar_augmentation, add_mask=False)

    def augment_sample(self, input, target, augmentation=None):
        if augmentation is None:
            augmentation = self.new_augmentation()
        input = input.unsqueeze(dim=0) # making a batch from an image
        point_cloud_x=[]
        point_cloud_y=[]
        segmentation_to_object = []
        for n, coco_object in enumerate(target):
            for surface_n in range(len(coco_object["segmentation"])):
                X = torch.tensor(coco_object["segmentation"][surface_n])[::2]
                Y = torch.tensor(coco_object["segmentation"][surface_n])[1::2]
                point_cloud_x.append(X)
                point_cloud_y.append(Y)
                segmentation_to_object.append(n)
        point_tensor_x = torch.zeros([1, len(point_cloud_x), max([len(x) for x in point_cloud_x])])
        point_tensor_y = torch.zeros_like(point_tensor_x)
        for n in range(len(point_cloud_x)):
            point_tensor_x[0, n, :len(point_cloud_x[n])]=point_cloud_x[n]
            point_tensor_y[0, n, :len(point_cloud_y[n])]=point_cloud_y[n]

        # If COCO provide matrices .... these would be the only 2 lines.
        out_image = augmentation(input)
        point_tensor_x, point_tensor_y = augmentation.forward_point_cloud((point_tensor_x, point_tensor_y), input)
        print("DBG 77",point_tensor_x.size())

        new_target = []
        for object_n, coco_object in enumerate(target):
            coco_object = {k:v for k, v in coco_object.items()} # shallow copy
            segmentation = []
            for segm_id in [n for n in range(len(segmentation_to_object)) if segmentation_to_object[n]==object_n]:
                print(point_tensor_x.size(), point_tensor_y.size())
                obj_x=point_tensor_x[0,segm_id,:len(point_cloud_x[segm_id])].view([-1,1])
                obj_y=point_tensor_y[0,segm_id,:len(point_cloud_x[segm_id])].view([-1,1])
                segm_xy=torch.cat([obj_x,obj_y], dim=1)
                segmentation.append(segm_xy.view(-1).tolist())
            coco_object["segmentation"] = segmentation
            all_object_x=torch.tensor([item for sublist in segmentation for item in sublist])[::2]
            all_object_y=torch.tensor([item for sublist in segmentation for item in sublist])[1::2]
            left=all_object_x.min().item()
            right=all_object_x.max().item()
            top=all_object_y.min().item()
            bottom=all_object_y.max().item()
            coco_object["bbox"] = [left, top, right-left, bottom-top]
            new_target.append(coco_object)
        return out_image[0, :, :, :], new_target

    def show_augmentation(self, n, augmentation=None, save_filename=None):
        if augmentation is None:
            augmentation = self.new_augmentation()
        simple_input, simple_target = self.ds[n]
        augmented_input, augmented_target = self.augment_sample(simple_input, simple_target, augmentation=augmentation)
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(simple_input.transpose(2,0).transpose(1,0))
        ax2.imshow(augmented_input.transpose(2,0).transpose(1,0))
        for coco_object in simple_target:
            x=np.array(coco_object["segmentation"][0][::2])
            y=np.array(coco_object["segmentation"][0][1::2])
            #ax1.fill(x,y,alpha=.3)
            ax1.plot(x,y)
            l,t,w,h=coco_object["bbox"]
            r, b = l+w, t+h
            #ax1.plot([l,r,r,l,l],[t,t,b,b,t])
            print(x,y)
        for coco_object in augmented_target:
            x=np.array(coco_object["segmentation"][0][::2])
            y=np.array(coco_object["segmentation"][0][1::2])
            print("->",x.astype("int32").tolist(),y.astype("int32").tolist())
            ax2.plot(x,y)
            l,t,w,h=coco_object["bbox"]
            r, b = l+w, t+h
            #ax2.plot([l,r,r,l,l],[t,t,b,b,t])

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
        elif isinstance(augmentation, ChannelImageAugmentation):
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
    def __init__(self, dataset, augmentation_factory, apply_on="guess",append_input_map=False):
        self.dataset = dataset
        self.augmentation_factory = augmentation_factory
        self.apply_on = _get_augmentation_subjects_from_dataset(dataset, where=apply_on, augmentation=augmentation_factory())
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
        self.apply_on = _get_augmentation_subjects_from_dataset(dataset, where=apply_on, augmentation=augmentations[0]())
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
