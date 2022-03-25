from .deterministic_image_augmentation import DeterministicImageAugmentation
from .augmentation_choice import AugmentationChoice
from .augmentation_cascade import AugmentationCascade

#def count_counterfactuals



class AugmentationCounterfactual(DeterministicImageAugmentation):
    @staticmethod
    def create(wrapped_augmentation, prob):
        if isinstance(wrapped_augmentation, AugmentationChoice):
            pass
        elif isinstance(wrapped_augmentation, AugmentationCascade):
            pass
        else:
            raise ValueError
    def __init__(self):
        pass

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        if self.is_choice:
            augmentation = type(self).available_augmentations[self.]()
        else:
            return self.wrapped.forward_img(batch_tensor)



    def forward_mask(self, batch_tensor: torch.Tensor) -> torch.LongTensor:
        pass
    def for





class AugmentatationCounterFactual(DeterministicImageAugmentation):
    def __init__(self, augmentation):
        #self.augmentations = [aug_cls() for aug_cls in type(self).augmentation_list]
        pass