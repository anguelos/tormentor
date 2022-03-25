from .random import Categorical
from .deterministic_image_augmentation import DeterministicImageAugmentation, SamplingField, PointCloudList, PointCloudsImages, random_fork
import torch


class AugmentationCascade(DeterministicImageAugmentation):
    r"""Select randomly among many augmentations.

    .. figure :: _static/example_images/AugmentationCascade.png

        Cascade of perspective augmentation followed by plasma-brightness

        .. code-block :: python

            augmentation_factory = tormentor.RandomPerspective | tormentor.RandomPlasmaBrightness

    A more complete usage of AugmentationCascade and AugmentationChoice can be seen in the following listing
    which produces the following computation graph. In the graph AugmentationCascade can be though of as all arrows
    that don't leave an AugmentationChoice

    .. code-block :: python

        from tormentor import RandomColorJitter, RandomFlip, RandomWrap, \
            RandomPlasmaBrightness, RandomPerspective, \
            RandomGaussianAdditiveNoise, RandomRotate

        linear_aug = (RandomFlip ^ RandomPerspective ^ RandomRotate)  | RandomColorJitter
        nonlinear_aug = RandomWrap | RandomPlasmaBrightness
        final_augmentation = (linear_aug ^ nonlinear_aug) | RandomGaussianAdditiveNoise

        epochs, batch_size, n_points, width, height = 10, 5, 20, 320, 240

        for _ in range(epochs):
            image_batch = torch.rand(batch_size, 3, height, width)
            segmentation_batch = torch.rand(batch_size, 1, height, width).round()
            augmentation = final_augmentation()
            augmented_images = augmentation(image_batch)
            augmented_gt = augmentation(segmentation_batch)
            # Train and do other things

    .. image:: _static/img/routing.svg


    """

    def __init__(self):
        super().__init__()
        self.augmentations = [aug_cls() for aug_cls in type(self).augmentation_list]

    def __call__(self, *args, **kwargs):
        current_args = args
        for augmentation in self.augmentations:
            current_args = augmentation(*current_args, **kwargs)
            if not isinstance(current_args, tuple):
                current_args = (current_args,)
        if isinstance(current_args, tuple) and len(current_args) == 1:
            return current_args[0]
        else:
            return current_args

    def augment_sampling_field(self, sf: SamplingField) -> SamplingField:
        device = sf[0].device
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                sf = augmentation.forward_sampling_field(sf)
        return sf

    def augment_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = image_tensor.device
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                image_tensor = augmentation.forward_img(image_tensor)
        return image_tensor

    def augment_mask(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = image_tensor.device
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                image_tensor = augmentation.forward_mask(image_tensor)
        return image_tensor

    def forward_img_path_probabilities(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        device = batch_tensor.device
        probs = torch.ones(batch_tensor.size(0), device=device)
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                probs = probs * augmentation.forward_img_path_probabillities(batch_tensor)
        return probs


    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        device = coords[0].device
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                coords = augmentation.forward_img(coords)
        return coords
        # raise NotImplemented  # determinism forbids running under other seed

    def forward_bboxes(self, bboxes: torch.FloatTensor, image_tensor=None, width_height=None) -> torch.FloatTensor:
        #device = bboxes.device
        #with random_fork(devices=(device,)):
        #    for augmentation in self.augmentations:
        #        torch.manual_seed(augmentation.seed)
        #        batch_tensor = augmentation.forward_img(batch_tensor)
        # return batch_tensor
        # TODO(anguelos) double check this, it is quite dangerous
        raise NotImplemented  # determinism forbids running under other seed

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        device = batch_tensor.device
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                batch_tensor = augmentation.forward_img(batch_tensor)
        return batch_tensor
        # raise NotImplemented  # determinism forbids running under other seed

    def forward_mask(self, batch_tensor: torch.LongTensor) -> torch.LongTensor:
        device = batch_tensor.device
        with random_fork(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                batch_tensor = augmentation.forward_mask(batch_tensor)
        return batch_tensor
        # raise NotImplemented  # determinism forbids running under other seed

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor,
                           compute_img: bool) -> PointCloudsImages:
        # TODO(anguelos) double check this, it is quite dangerous
        raise NotImplemented  # determinism forbids running under other seed

    @classmethod
    def create(cls, augmentation_list):
        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000, 9000000, (1,)).item()}"
        new_cls = type(new_cls_name, (cls,), {"augmentation_list": augmentation_list,
                                              "aumentation_instance_list": [aug() for aug in augmentation_list]})
        return new_cls

    @classmethod
    def get_distributions(cls, copy: bool = True):
        res = {}
        for n, contained_augmentation in enumerate(cls.augmentation_list):
            aug_name = f"{contained_augmentation.__qualname__}{n}"
            res.update({f"{aug_name}: {k}": v for k, v in contained_augmentation.get_distributions(copy=copy).items()})
        return res
