import torch
from .sampling_fileds import SamplingField, apply_sampling_field, create_sampling_field
from .deterministic_image_augmentation import DeterministicImageAugmentation


class SpatialImageAugmentation(DeterministicImageAugmentation):
    r"""Parent class for augmentations that move things around.

    Every class were image pixels move around rather that just change should be a descendant of this class.
    All classes that do not descend from this class are expected to be neutral for pointclouds, and sampling fields and
    should be descendants of StaticImageAugmentation.
    """

    @classmethod
    def functional_sampling_field(cls, coords: SamplingField, *state) -> SamplingField:
        raise NotImplementedError()

    def forward_img(self, batch_tensor):
        batch_size, channels, height, width = batch_tensor.size()
        sf = create_sampling_field(width, height, batch_size=batch_size, device=batch_tensor.device)
        sf = self.forward_sampling_field(sf)
        return apply_sampling_field(batch_tensor, sf)

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        state = self.generate_batch_state(coords)
        return type(self).functional_sampling_field(*((coords,) + state))

    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_img(X)

