import torch
from .base_augmentation import DeterministicImageAugmentation, AugmentationChoice, AugmentationCascade


class AugmentationFactory(torch.nn.Module):
    """Class that wraps augmentation classes

    Each augmentation type is supposed to be immutable so changing the random distributions of it is realised through
    automatic subclassing. The main utility of this class is to hide the automatic subclassing from the user and present
    Augmentation classes as objects.
    As augmentations can be employed as regularisation layers, this class extends torch.nn.Module.

    """
    def custom(self, **kwargs):
        new_augmentation = self.augmentation_cls.override_distributions(**kwargs)
        return AugmentationFactory(new_augmentation)

    def __instancecheck__(self, instance: object):
        return isinstance(instance, self.augmentation_cls)

    def get_distributions(self, copy=True):
        # wraps DeterministicImageAugmentation classmethod
        return self.augmentation_cls.get_distributions(copy=copy)


    def override_distributions(self, **kwargs):
        # wraps DeterministicImageAugmentation classmethod
        # but mutates self
        new_augmentation_cls = self.augmentation_cls.override_distributions(**kwargs)
        self.augmentation_cls = new_augmentation_cls
        return self


    def __init__(self, augmentation_cls):
        assert issubclass(augmentation_cls,DeterministicImageAugmentation)
        self.augmentation_cls = augmentation_cls


    def __or__(self, other):
        if issubclass(other, DeterministicImageAugmentation):
            other_augmentation_cls = other
        elif isinstance(other, AugmentationFactory):
            other_augmentation_cls = other.augmentation_cls
        else:
            raise ValueError("operator only defined for augmentations or their factories")

        if issubclass(other_augmentation_cls, AugmentationChoice):
            other_augmentation_cls_list = list(other_augmentation_cls.available_augmentations)
        else:
            other_augmentation_cls_list = [other_augmentation_cls]

        if issubclass(self.augmentation_cls, AugmentationChoice):
            my_augmentation_cls_list = list(self.augmentation_cls.available_augmentations)
        else:
            my_augmentation_cls_list = [self.augmentation_cls]

        choice = AugmentationChoice.create(augmentation_list=my_augmentation_cls_list + other_augmentation_cls_list)
        return AugmentationFactory(choice)

    def __rxor__(self, other):
        print("XOR")
        return AugmentationFactory._choose(other, self)

    def __xor__(self, other):
        print("XOR")
        return AugmentationFactory._choose(self, other)

    def __and__(self, other):
        return AugmentationFactory._concat(self, other)

    def __and__(self, other):
        return AugmentationFactory._concat(other, self)

    @staticmethod
    def _choose(left, right):
        if isinstance(left, type) and issubclass(left, DeterministicImageAugmentation):
            left_augmentation_cls = left
        elif isinstance(left, AugmentationFactory):
            left_augmentation_cls = left.augmentation_cls
        else:
            raise ValueError("operator only defined for augmentations or their factories")

        if isinstance(right, type) and issubclass(right, DeterministicImageAugmentation):
            right_augmentation_cls = right
        elif isinstance(right, AugmentationFactory):
            right_augmentation_cls = right.augmentation_cls
        else:
            raise ValueError("operator only defined for augmentations or their factories")

        if issubclass(left_augmentation_cls, AugmentationChoice):
            left_augmentation_cls_list = list(left_augmentation_cls.available_augmentations)
        else:
            left_augmentation_cls_list = [left_augmentation_cls]

        if issubclass(right_augmentation_cls, AugmentationChoice):
            right_augmentation_cls_list = list(right_augmentation_cls.available_augmentations)
        else:
            right_augmentation_cls_list = [right_augmentation_cls]

        choice = AugmentationChoice.create(augmentation_list=left_augmentation_cls_list + right_augmentation_cls_list)
        return AugmentationFactory(choice)



    @staticmethod
    def _concat(left, right):
        if isinstance(right, type) and issubclass(right, DeterministicImageAugmentation):
            right_augmentation_cls = right
        elif isinstance(right, AugmentationFactory):
            right_augmentation_cls = right.augmentation_cls
        else:
            raise ValueError("operator only defined for augmentations or their factories")

        if isinstance(left, type) and issubclass(left, DeterministicImageAugmentation):
            left_augmentation_cls = left
        elif isinstance(right, AugmentationFactory):
            left_augmentation_cls = left.augmentation_cls
        else:
            raise ValueError("operator only defined for augmentations or their factories")

        if issubclass(right_augmentation_cls, AugmentationCascade):
            other_augmentation_cls_list = list(right_augmentation_cls.augmentation_list)
        else:
            other_augmentation_cls_list = [right_augmentation_cls]

        if issubclass(left_augmentation_cls, AugmentationCascade):
            my_augmentation_cls_list = list(left_augmentation_cls.augmentation_list)
        else:
            my_augmentation_cls_list = [left_augmentation_cls]

        cascade = AugmentationCascade.create(augmentation_list=my_augmentation_cls_list + other_augmentation_cls_list)
        return AugmentationFactory(cascade)

    def __call__(self, *args, **kwargs):
        return self.augmentation_cls(*args, **kwargs)

    def __repr__(self):
        if issubclass(self.augmentation_cls, AugmentationChoice):
            children = [repr(AugmentationFactory(aug_cls)) for aug_cls in self.augmentation_cls.available_augmentations]
            children = sorted(children)
            return "(" + " ^ ".join(children) + ")"
        elif issubclass(self.augmentation_cls, AugmentationCascade):
            children = [repr(AugmentationFactory(aug_cls)) for aug_cls in self.augmentation_cls.augmentation_list]
            children = reversed(children)
            return "(" + " & ".join(children) + ")"
        else:
            dist_str = ", ".join([f"{name}={str(dist)}" for name, dist in self.get_distributions().items()])
            return f"Random{self.augmentation_cls.augmentation_type().__qualname__}.custom({dist_str})"

    def __eq__(self, other):
        return repr(self) == repr(other)
