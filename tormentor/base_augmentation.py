import torch
from itertools import count
import types
import kornia as K
from .random import Distribution, Categorical
from typing import Tuple, Union, List

SamplingField = Tuple[torch.FloatTensor, torch.FloatTensor]
PointCloud = Tuple[torch.FloatTensor, torch.FloatTensor]
PointCloudList = List[PointCloud]
PointCloudsImages = Tuple[PointCloudList, torch.FloatTensor]
PointCloudOneOrMore = Union[PointCloudList, PointCloud]
AugmentationState = Tuple[torch.Tensor, ...]

# This is a work around probably wrong behavior of pytorch
# A bug report has been submitted to pytorch
# https://github.com/pytorch/pytorch/issues/41970
if torch.cuda.device_count() == 0:
    def random_fork(devices):
        return torch.random.fork_rng()
else:
    def random_fork(devices):
        return torch.random.fork_rng(devices=devices)


def _use_sampling_field(sf: SamplingField):
    return None

def is_sampling_field(var):
    r"""Returns True is var is a SamplingField.

    A tuple containing two torch.FloatTensors

    Args:
        var:

    Returns:

    """
    try:
        _use_sampling_field(var)
        return True
    except TypeError:
        return False


def is_typing(var, type_definition):
    r"""Returns True is var is a type definition.

    As oposed to instanceof, this works with any type used as a typing hint.

    Args:
        var:

    Returns:

    """
    def _f(parameter: type_definition) -> int:
        return parameter
    try:
        _f(var)
        return True
    except TypeError:
        return False


def create_sampling_field(width: int, height: int, batch_size: int = 1, device: torch.device = "cpu") -> SamplingField:
    r"""Creates a SamplingField.

    A SamplingField is a tuple of 3D tensors of the same size. Sampling fields are augmentable by all augmentations
    although many augmentations (Non-spatial) have no effect on them. The can be used to resample images, pointclouds,
    masks. When sampling, for both axes, the input image is interpreted to lie on the region [-1, 1]. The output image
    when resampling will have the width and height of the sampling field. A sampling field can also refer to a single
    image rather than a batch in whitch case the tensors are 2D.
    The first dimension is the batch size.
    The second dimension is the width of the output image after sampling.
    The third dimension is the width of the output image after sampling.
    The created sampling fields are normalised in the range [-1,1] regardless of their size.
    Although not enforced, it is expected that augmentations are homomorphisms.
    Sampling fields are expected to operate identically on all channels and dont have a channel dimension.

    Args:
        width: The sampling fields width.
        height:  The sampling fields height.
        batch_size: If 0, the sampling field refers to a single image. Otherwise the first dimension of the tensors.
            Created sampling fileds are simply repeated over the batch dimension. Default value is 1.
        device: the device on which the sampling filed will be created.

    Returns:
        A tuple of 3D or 2D tensors with values ranged in [-1,1]

    """
    sf = K.utils.create_meshgrid(height=height, width=width, normalized_coordinates=True, device=device)
    sf = (sf[:, :, :, 0], sf[:, :, :, 1])
    if batch_size == 0:
        return sf[0][0, :, :], sf[1][0, :, :]
    else:
        return sf[0].repeat([batch_size, 1, 1]), sf[1].repeat([batch_size, 1, 1])


def apply_sampling_field(input_img: torch.Tensor, coords: SamplingField):
    r"""Resamples one or more images by applying sampling fields.

    Bilinear interpolation is employed.

    Args:
        input_img: A 4D float tensor [batch x channel x height x width] or a 3D tensor [channel x height x width].
            Containing the image or batch from which the image is sampled.
        coords: A sampling field with 3D [batch x out_height x out_width] or 2D [out_height x out_width]. The dimensions
            of the sampling fields must be one less that the input_img.

    Returns:
        A tensor of as many dimensions [batch x channel x out_height x out_width] or [channel x out_height x out_width]]
        as the input.
    """
    x_coords, y_coords = coords
    if input_img.ndim == 3:
        assert coords[0].ndim == 2
        x_coords, y_coords = x_coords.unsqueeze(dim=0), y_coords.unsqueeze(dim=0)
        batch = input_img.unsqueeze(dim=0)
    else:
        batch = input_img
    xy_coords = torch.cat((x_coords.unsqueeze(dim=-1), y_coords.unsqueeze(dim=-1)), dim=3)
    sampled_batch = torch.nn.functional.grid_sample(batch, xy_coords, align_corners=True)
    if input_img.ndim == 3:
        return sampled_batch[0, :, :, :]
    else:
        return sampled_batch


class AugmentationAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, random_parametrers):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class DeterministicImageAugmentation(object):
    """Deterministic augmentation functor and its factory.

    This class is the base class realising an augmentation.
    In order to create a create_persistent augmentation the forward_sample_img method and the factory class-function
    have to be defined.
    If a constructor is defined it must call ``super().__init__(**kwargs)`` .

    The ``**kwargs`` on the constructor define all needed variables for generating random augmentations.
    The factory method returns a lambda that instanciates augmentatations.
    Any parameter about the augmentations randomness should be passed by the factory to the constructor and used inside
    forward_sample_img's definition."""

    _ids = count(0)
    _global_seed = torch.LongTensor(1).random_(1000000000, 2000000000).item()
    distributions = {}

    @staticmethod
    def reset_all_seeds(global_seed=None):
        DeterministicImageAugmentation._ids = count(0)
        if global_seed is None:
            global_seed = 0
        DeterministicImageAugmentation._global_seed = global_seed

    def __init__(self, aug_id=None, seed=None):
        if aug_id is None:
            self.aug_id = next(DeterministicImageAugmentation._ids)
        else:
            self.aug_id = aug_id
        if seed is None:
            self.seed = self.aug_id + DeterministicImageAugmentation._global_seed
        else:
            self.seed = seed

    def like_me(self):
        r"""Returns a new augmentation following the same distributions as self.

        Returns:
            An instance of the same class as self.
        """
        return type(self)()

    def augment_image(self, image_tensor: torch.FloatTensor):
        r"""Augments an image or a batch of images.

        This method enforces determinism for image data. Only the batch dimension is guarantied to be preserved.
        Channels, width, and height dimensions can differ on the outputs. This method should be treated as final and
        should not be redefined in subclasses. All subclasses should implement ``forward_image`` instead. Images can
        have any number of channels although some augmentations eg. those manipulating the color-space might
        expect specific channel counts.

        Args:
            image_tensor: a float tensor of [batch x channels x height x width] or [channels x height x width].

        Returns:
            An image or a batch of tensors sized [batch x channels x height x width] or [channels x height x width]"""
        device = image_tensor.device

        with random_fork(devices=(device,)):
            torch.manual_seed(self.seed)
            n_dims = len(image_tensor.size())
            if n_dims == 3:
                return self.forward_img(image_tensor.unsqueeze(dim=0))[0, :, :, :]
            elif n_dims == 4:
                return self.forward_img(image_tensor)
            else:
                raise ValueError("image_tensor must represent a sample or a batch")

    def augment_mask(self, mask_tensor: torch.Tensor):
        r"""Augments an mask or a batch of masks.

        Masks differ from images as they are interpreted to answer a pixel-wise "where" question. Although technically
        they can be indistinguishable from images they are interpreted differently. A typical example would be a
        dense segmentation mask such containing class one-hot encoding along the channel dimension.
        This method enforces determinism for mask data. Only the batch dimension is guarantied to be preserved.
        Channels, width, and height dimensions can differ on the outputs. This method should be treated as final and
        should not be redefined in subclasses. A subclasses should implement ``forward_mask`` instead.

        Args:
            mask_tensor: a float tensor of [batch x channels x height x width] or [channels x height x width].

        Returns:
            An mask or a batch of tensors sized [batch x channels x height x width] or [channels x height x width]"""
        device = mask_tensor.device
        with random_fork(devices=(device,)):
            torch.manual_seed(self.seed)
            n_dims = len(mask_tensor.size())
            # TODO(anguelos) allow for class index long tensors
            # TODO(anguelos) allow to enforce probabilistic interpretation of channels
            if mask_tensor.dtype == torch.bool:
                float_mask_tensor = mask_tensor.float()
            else:
                float_mask_tensor = mask_tensor
            if n_dims == 3:
                res = self.forward_mask(float_mask_tensor.unsqueeze(dim=0))[0, :, :]
            elif n_dims == 4:
                res = self.forward_mask(float_mask_tensor)
            else:
                raise ValueError("mask_tensor must represent a sample [CxHxW] or a batch [BxCxHxW]")
            if mask_tensor.dtype == torch.bool:
                res = res > .5
            return res

    def augment_sampling_field(self, sf: SamplingField):
        r"""Augments a sampling field for an image or samplingfileds for batches.

        Sampling fields are the way to see how augmentations move things around. A sampling field can be generated with
        ``create_sampling_field`` and be used to resample image data with ``apply_sampling_field``

        This method enforces determinism for image data. Only the batch dimension is guarantied to be preserved.
        Channels, width, and height dimensions can differ on the outputs. This method should be treated as final and
        should not be redefined in subclasses. A subclasses should implement ``forward_sampling_field`` instead.

        Args:
            sf: a tuple with 2 float tensors of the same size. Either [batch x height x width] or [height x width]

        Returns:
            A tuple of 2 tensors sized [batch x new_height x new_width] or [new_height x new_width]"""
        assert sf[0].size() == sf[1].size()
        device = sf[0].device
        with random_fork(devices=(device,)):
            torch.manual_seed(self.seed)
            n_dims = len(sf[0].size())
            if n_dims == 2:
                sf = sf[0].unsqueeze(dim=0), sf[1].unsqueeze(dim=0)
                sf = self.forward_sampling_field(sf)
                return sf[0][0, :, :], sf[1][0, :, :]
            elif n_dims == 3:
                return self.forward_sampling_field(sf)
            else:
                raise ValueError("sampling fields must represent a sample or a batch")

    def augment_pointcloud(self, pc: PointCloudOneOrMore, image_tensor: torch.FloatTensor, compute_img: bool):
        r"""Augments pointclouds over an image or a batch.

        Pointclouds are defined to be in pixel coordinates in contextualised by an image or at least an image size.
        The pointcloud for a single image is a tuple of 1D float tensors (vectors) one with the X coordinates and one
        with the Y coordinates. If the image_tensor is a batch, then a list of pointclouds is associated with the batch,
        one for every image in the batch. Pointcloud augmentation shares a lot of the heavy computation with augmenting
        its reference image tensor, both are employing an augmented sampling field. This method should be treated as
        final and should not be redefined in subclasses. A subclasses should implement ``forward_pointcloud`` instead.

        Args:
            pc: Either a tuple of vectors with X Y coordinates, or a list of many such tuples.
            image_tensor: A 3D tensor [channel x height x width] or a 4D tensor [batch x channel x height x width].
            compute_img: If True, the reference image will be augmented and returned, if false the reference image will
                be returned unaugmented.

        Returns:
            A tuple with a pointcloud or a list of pointclouds, and a 3D or 4D tensor. The image tensor is either the
            original ``image_tensor`` or the same exact augmentation applied the point cloud.
        """
        if isinstance(pc, list):
            device = pc[0][0].device
        else:
            device = pc[0].device
        with random_fork(devices=(device,)):
            torch.manual_seed(self.seed)
            if isinstance(pc, list):
                assert image_tensor.ndim == 4
                out_pc, out_img = self.forward_pointcloud(pc, image_tensor, compute_img=compute_img)
                return out_pc, out_img
            elif isinstance(pc[0], torch.Tensor):
                assert image_tensor.ndim == 3
                out_pc, out_img = self.forward_pointcloud([pc], image_tensor.unsqueeze(dim=0), compute_img=compute_img)
                out_pc = out_pc[0]
                out_img = out_img[0, :, :, :]
                return out_pc, out_img
            else:
                raise ValueError("point clouds must represent a sample or a batch")

    def __call__(self, *args, compute_img: bool = True, is_mask: bool = False):
        r"""This method routes to the apropriate "augment" method depending on *args.

        In essence __call__ implements method overloading for different kinds of data and simply routes to the
        apropriate method. If it is preffered to state the routing explicitly, the "self.augment_..." methods should be
        used. Pointclouds as sampling_fields.

        In what concersn mask augmentations:
        When masks are augmented either as label tensors or onehots, class zero is chosen as a tiebreaker. augmented
        onehots are softly rounded by beeing scaled and then having a softmax aplied along the channel dimension.

        Args:
            *args: can be one of the following
                [Pointcloud, image]: calls ``self.augment_pointcloud``
                [List[Pointcloud], batch]: calls ``self.augment_pointcloud``

                [SamplingFiled]: calls ``self.augment_sampling_field``

                [torch.Tensor]: calls ``self.augment_sampling_image`` or calls ``self.augment_mask`` if the tensor has
                    a discrete dtype it is intepreted to be [batch x height x width] or [height x width] converted to a
                    onehot encoding as the channel and augmented as a mask, this includes dtype troch.bool.

            compute_img: if True, the image_data contextualising the pointcloud will also be augmented and they will
                share the sampling field computation they both use.
            is_mask: if True, the image tensor is considered a mask. If args[0] is a discrete Tensor, this flag is
                ignored and ``self.augment_mask`` is called.

        Returns:
            data equivalent what was passed in args"""
        # practiaclly method overloading
        # if I could only test objects for Typing Generics
        assert len(args) == 1 or len(args) == 2
        if len(args) == 1  and isinstance(args[0], torch.Tensor) and args[0].dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            assert 2 <= args[0].ndim <= 3 # labels are to be expanded as the channel dimesion
            n_channels, height, width = args[0].max() + 1, args[0].size(-2), args[0].size(-1)
            if args[0].ndim == 2:
                batch_sz = 1
                input_batch = args[0].unsqueeze(dim=0)
            else:
                batch_sz = args[0].size(0)
                input_batch = args[0]
            batch_onehots = torch.empty([batch_sz, n_channels, height, width], dtype=torch.float,
                                                device=input_batch.device)
            augmented_onehot_mask = torch.empty([batch_sz, n_channels, height, width], dtype=torch.float, device=input_batch.device)
            for channel in range(n_channels):  # todo(anguelos) implement a better onehot
                batch_onehots[:, channel, :, :] = (input_batch[:, :, :] == channel).float()
                binary_slice = (input_batch[:, :, :] == channel).unsqueeze(dim=1).float()
                augmented_onehot_mask[:, channel: channel + 1, :, :] = self.augment_mask(binary_slice)
            #augmented_onehot_mask = self.augment_mask(batch_onehots)

            epsilon = torch.zeros([1, n_channels, 1, 1], device=augmented_onehot_mask.device)
            epsilon[0, 0, 0, 0] = .00000000001 # lets favor the zero class if a pixels label is ambiguous.
            # todo(anguelos) should we allow the enduser control on the tie-breaker class?

            batch_labels = torch.argmax(augmented_onehot_mask + epsilon, dim=1)
            batch_labels = batch_labels.to(args[0].dtype)
            if args[0].ndim == 2:
                return batch_labels[0, :, :]
            else:
                return batch_labels
        elif len(args) == 2:  # pointcloud and image tensor
            pointcloud, image_tensor = args
            return self.augment_pointcloud(pointcloud, image_tensor, compute_img)
        elif isinstance(args[0], tuple):  # sampling field
            assert (2 <= args[0][0].ndim <= 3)
            return self.augment_sampling_field(args[0])
        elif isinstance(args[0], torch.Tensor) and not is_mask:  # image
            assert 3 <= args[0].ndim <= 4
            return self.augment_image(args[0])
        elif isinstance(args[0], torch.Tensor) and is_mask:
            assert 3 <= args[0].ndim <= 4
            resulting_mask = self.augment_mask(args[0])
            if resulting_mask.size(-3) > 1: # making sure the mask preserves a onehot-like probabillity
                if resulting_mask.ndim == 4:
                    epsilon = torch.zeros([1, resulting_mask.size(1), 1, 1], device=resulting_mask.device)
                    epsilon[0, 0, 0, 0] = .01  # lets favor the zero class if a pixels label is ambiguous.
                    # todo(anguelos) should we allow the enduser control on the tie-breaker class?
                    resulting_mask = torch.softmax((resulting_mask + epsilon) * 1000, dim=1)
                else:
                    epsilon = torch.zeros([resulting_mask.size(0), 1, 1], device=resulting_mask.device)
                    epsilon[0, 0, 0] = .01  # lets favor the zero class if a pixels label is ambiguous.
                    # todo(anguelos) should we allow the enduser control on the tie-breaker class?
                    resulting_mask = torch.softmax((resulting_mask + epsilon) * 1000, dim=0)
            return resulting_mask
        else:
            raise ValueError

    def __repr__(self):
        param_names = ("aug_id", "seed")
        param_assignments = ", ".join(["{}={}".format(k, repr(self.__dict__[k])) for k in param_names])
        return self.__class__.__qualname__ + "(" + param_assignments + ")"

    def __eq__(self, obj):
        return self.__class__ is obj.__class__ and self.aug_id == obj.aug_id and self.seed == obj.seed

    def __str__(self):
        functions = (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType)
        attribute_strings = []
        for name in self.__dict__.keys():
            attribute = getattr(self, name)
            if (not isinstance(attribute, functions)) and (not name.startswith("__")):
                attribute_strings.append(f"{name}:{repr(attribute)}")
        return self.__class__.__qualname__ + ":\n\t" + "\n\t".join(attribute_strings)

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> AugmentationState:
        """Generates deterministic state for each augmentation.

        Returns: a tuple of tensors representing the complete state of the augmentaion so that a

        """
        raise NotImplementedError()

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        """Distorts a batch of one or more images.

        :param batch_tensor: Images are 4D tensors of [batch_size, #channels, height, width] size.
        :return: A create_persistent 4D tensor [batch_size, #channels, height, width] with the create_persistent image.
        """
        raise NotImplementedError()

    def forward_mask(self, batch_tensor: torch.Tensor) -> torch.LongTensor:
        """Distorts a a batch of one or more masks.

        :param batch_tensor: Images are 4D tensors of [batch_size, #channels, height, width] size.
        :return: A create_persistent 4D tensor [batch_size, #channels, height, width] with the create_persistent image.
        """
        raise NotImplementedError()

    @classmethod
    def get_distributions(cls, copy=True):
        res = {}
        for me_or_parent in reversed(cls.mro()):
            if copy:
                res.update({k: v.copy() for k, v in me_or_parent.__dict__.items() if isinstance(v, Distribution)})
            else:
                res.update({k: v for k, v in me_or_parent.__dict__.items() if isinstance(v, Distribution)})
        return res

    @classmethod
    def augmentation_type(cls):
        try:
            return cls.original_augmentation
        except AttributeError:
            return cls

    @classmethod
    def override_distributions(cls, requires_grad=False, **kwargs):
        cls_members = cls.get_distributions()
        assert all([isinstance(v, Distribution) for v in kwargs.values()])
        assert set(kwargs.keys()) <= set(cls_members.keys())
        cls_members.update(kwargs)
        for cls_distribution in cls_members.values():
            for parameter in cls_distribution.get_distribution_parameters().values():
                parameter.requires_grad_(requires_grad)
        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000, 9000000, (1,)).item()}"

        cls_members.update({"original_augmentation": cls.augmentation_type()})

        new_cls = type(new_cls_name, (cls,), cls_members)
        return new_cls

    def forward_bboxes(self, bboxes: torch.FloatTensor, image_tensor=None, width_height=None) -> torch.FloatTensor:
        """Applies a transformation on Image coordinate defined bounding boxes.

        Bounding Boxes are encoded as [Left, Top, Right, Bottom]

        Args:
            bboxes (torch.FloatTensor) : A tensor with bboxes for a sample [N x 4] or a batch [S x N x 4]
            image_tensor (torch.FloatTensor): A valid batch image tensor [S x C x H x C] or sample image tensor
                [C x H x W]. In both cases it only used to normalise bbox coordinates and can be omitted if width_height
                is specified.
            width_height (int, int ): Values used to normalise bbox coordinates to [-1,1] and back, should be ommited if
                image tensor is passed

        Returns: a tensor with the bounding boxes of the transformed bounding box.


        """
        """Applies a transformation on Image coordinate defined bounding boxes.

        Bounding Boxes are encoded as [Left, Top, Right, Bottom]

        Args:
            bboxes (torch.FloatTensor) : A tensor with bboxes for a sample [N x 4] or a batch [S x N x 4]
            image_tensor (torch.FloatTensor): A valid batch image tensor [S x C x W x H] or sample image tensor
                [C x H x W]. In both cases it only used to normalise bbox coordinates and can be omitted if width_height
                is specified.
            width_height (int, int ): Values used to normalise bbox coordinates to [-1,1] and back, should be ommited if
                image tensor is passed

        Returns: a tensor with the bounding boxes of the transformed bounding box.


        """
        if len(bboxes.size()) == 2:
            bboxes = bboxes.unsqueeze(dim=0)
        if image_tensor is not None:
            width = image_tensor.size(-1)
            height = image_tensor.size(-2)
        else:
            width, height = width_height
        normalise_bboxes = torch.tensor([[width * .5, height * .5, width * .5, height * .5]])
        normalised_bboxes = bboxes / normalise_bboxes - 1
        bboxes_left = normalised_bboxes[:, 0].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_top = normalised_bboxes[:, 1].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_right = normalised_bboxes[:, 2].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_bottom = normalised_bboxes[:, 3].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_x = torch.concat([bboxes_left, bboxes_right, bboxes_right, bboxes_left], dim=1)
        bboxes_y = torch.concat([bboxes_top, bboxes_top, bboxes_bottom, bboxes_bottom], dim=1)

        pointcloud = (bboxes_x, bboxes_y)
        pointcloud = self.forward_pointcloud(pointcloud)
        pointcloud = torch.clamp(pointcloud[0], -1, 1), torch.clamp(pointcloud[0], -1, 1)

        left = ((pointcloud[0].min(dim=1) + 1) * .5 * width).view(-[1, 1])
        right = ((pointcloud[0].max(dim=1) + 1) * .5 * width).view(-[1, 1])
        top = ((pointcloud[1].min(dim=1) + 1) * .5 * height).view(-[1, 1])
        bottom = ((pointcloud[1].max(dim=1) + 1) * .5 * height).view(-[1, 1])
        result_bboxes = torch.concat([left, top, right, bottom], dim=1)
        return result_bboxes

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor,
                           compute_img: bool) -> PointCloudsImages:
        r"""Applies a transformation on normalised coordinate points.

        :param pcl, a pointcloud for every image in the batch pointclouds are given in pixel coordinates. The list must
        have the same size as the ``batch_tensor``. Each pointcloud is a tuple consisting of the vectors

        :type  pcl: list

        :param batch_tensor: The images to which each of the pointclouds refers [BxCxHxW]
        :type batch_tensor: torch.FloatTensor

        :param compute_img: If `False` the only the pointcloud will be computed, if `True` the images in the batch_tensor
         will also be augmented.
        :type compute_img: bool

        :return: augmented pointclouds and input image tensor or the augmented image tensor depending on compute_img.
        :rtype: PointCloudsImages
        """
        batch_sz, _, height, width = batch_tensor.size()
        sampling_field_tensor = K.utils.create_meshgrid(height, width, normalized_coordinates=True,
                                                        device=batch_tensor.device)
        sampling_field_tensor = sampling_field_tensor.repeat(batch_sz, 1, 1, 1)
        in_sampling_field = sampling_field_tensor[:, :, :, 0], sampling_field_tensor[:, :, :, 1]
        out_sampling_field = self.forward_sampling_field(in_sampling_field)

        out_pcl = []
        for batch_n in range(batch_sz):
            pc = pcl[batch_n]
            pc_normalised = pc[0] / (width * .5) - 1, pc[1] / (height * .5) - 1
            X_dst, Y_dst = out_sampling_field[0][batch_n, :, :].view(-1), out_sampling_field[1][batch_n, :, :].view(-1)
            X_src, Y_src = in_sampling_field[0][batch_n, :, :].view(-1), in_sampling_field[1][batch_n, :, :].view(-1)
            nearest_neighbor_src_X = torch.empty_like(pc_normalised[0])
            nearest_neighbor_src_Y = torch.empty_like(pc_normalised[1])
            # TODO(anguelos) should we indirectly allow control over gpu
            step = 10000000 // (width * height)  # This is about GPU memory
            for n in range(0, pc_normalised[0].size(0), step):
                pc_x, pc_y = pc_normalised[0][n: n + step], pc_normalised[1][n:n + step]
                euclidean = ((X_dst.view(1, -1) - pc_x.view(-1, 1)) ** 2 + (Y_dst.view(1, -1) - pc_y.view(-1, 1)) ** 2)
                idx = torch.argmin(euclidean, dim=1)
                nearest_neighbor_src_X[n:n + step] = X_src[idx][:]
                nearest_neighbor_src_Y[n:n + step] = Y_src[idx]
            out_pc = (nearest_neighbor_src_X + 1) * .5 * width, (nearest_neighbor_src_Y + 1) * .5 * height
            out_pcl.append(out_pc)
        if compute_img:
            out_coords = torch.cat((out_sampling_field[0].unsqueeze(dim=-1), out_sampling_field[1].unsqueeze(dim=-1)),
                                   dim=3)
            augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, out_coords, align_corners=True)
            return out_pcl, augmented_batch_tensor
        else:
            return out_pcl, batch_tensor

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        r"""Defines a spatial transform.

        Args:
            coords (): a tuple with two 3D float tensors each having a size of [Batch x Height x Width]. X and Y
                coordinates are in reference the range [-1, 1].

        Returns:
            The augmented samplint field.

        """
        raise NotImplementedError()


class StaticImageAugmentation(DeterministicImageAugmentation):
    r"""Parent class for augmentations that don't move things around.

    All classes that do descend from this are expected to be neutral for pointclouds, sampling fields although they
    might be erasing regions of the images so they are not gurantied to be neutral to masks.
    Every class were image pixels move around rather that just stay static, should be a descendant of
    SpatialImageAugmentation.
    """

    @classmethod
    def functional_image(cls, image_tensor, *state):
        raise NotImplementedError()

    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        return coords

    def forward_bboxes(self, bboxes: torch.FloatTensor, image_tensor=None, width_height=None):
        return bboxes

    def forward_pointcloud(self, pc: PointCloud, batch_tensor: torch.FloatTensor, compute_img: bool) -> PointCloud:
        if compute_img:
            return pc, self.forward_img(batch_tensor)
        else:
            return pc, batch_tensor

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        state = self.generate_batch_state(batch_tensor)
        return type(self).functional_image(*((batch_tensor,) + state))


class Identity(StaticImageAugmentation):
    def generate_batch_state(self, batch_tensor: torch.Tensor) -> AugmentationState:
        return ()

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        return batch_tensor


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

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        return NotImplemented  # determinism forbids running under other seed

    def forward_bboxes(self, bboxes: torch.FloatTensor, image_tensor=None, width_height=None) -> torch.FloatTensor:
        return NotImplemented  # determinism forbids running under other seed

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        return NotImplemented  # determinism forbids running under other seed

    def forward_mask(self, batch_tensor: torch.LongTensor) -> torch.LongTensor:
        return NotImplemented  # determinism forbids running under other seed

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor,
                           compute_img: bool) -> PointCloudsImages:
        return NotImplemented  # determinism forbids running under other seed

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


class AugmentationChoice(DeterministicImageAugmentation):
    r"""Select randomly among many augmentations.

    .. figure :: _static/example_images/AugmentationChoice.png

        Random choice of perspective and plasma-brightness augmentations

        .. code-block :: python

            augmentation_factory = tormentor.RandomPerspective ^ tormentor.RandomPlasmaBrightness
            augmentation = augmentation_factory()
            augmented_image = augmentation(image)
    """

    @classmethod
    def create(cls, augmentation_list, requires_grad=False):
        new_parameters = {"choice": Categorical(len(augmentation_list)), "available_augmentations": augmentation_list}
        for augmentation in augmentation_list:
            class_name = str(augmentation).split(".")[-1][:-2]
            cls_distributions = augmentation.get_distributions()
            cls_distributions = {f"{class_name}_{k}": v for k, v in cls_distributions.items()}
            new_parameters.update(cls_distributions)
        for cls_distribution in cls_distributions.values():
            for parameter in cls_distribution.get_distribution_parameters().values():
                parameter.requires_grad_(requires_grad)
        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000, 9000000, (1,)).item()}"
        new_cls = type(new_cls_name, (cls,), new_parameters)
        return new_cls

    def forward_sampling_field(self, coords: SamplingField):
        batch_sz = coords[0].size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch_x = []
        augmented_batch_y = []
        for sample_n in range(batch_sz):
            sample_coords = coords[0][sample_n: sample_n + 1, :, :], coords[1][sample_n: sample_n + 1, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            sample_x, sample_y = augmentation.forward_sampling_field(sample_coords)
            augmented_batch_x.append(sample_x)
            augmented_batch_y.append(sample_y)
        augmented_batch_x = torch.cat(augmented_batch_x, dim=0)
        augmented_batch_y = torch.cat(augmented_batch_y, dim=0)
        return augmented_batch_x, augmented_batch_y

    def forward_img(self, batch_tensor):
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            augmented_sample = augmentation.forward_img(sample_tensor)
            augmented_batch.append(augmented_sample)
        augmented_batch = torch.cat(augmented_batch, dim=0)
        return augmented_batch

    def forward_mask(self, batch_tensor):
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            augmented_sample = augmentation.forward_mask(sample_tensor)
            augmented_batch.append(augmented_sample)
        augmented_batch = torch.cat(augmented_batch, dim=0)
        return augmented_batch

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor,
                           compute_img: bool) -> PointCloudsImages:
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        augmented_pcl = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            pc_onelist = pcl[sample_n: sample_n + 1]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            aug_pc_onelist, augmented_sample = augmentation.forward_pointcloud(pc_onelist, sample_tensor, compute_img)
            augmented_batch.append(augmented_sample)
            augmented_pcl = augmented_pcl + aug_pc_onelist
        if compute_img:
            augmented_batch = torch.cat(augmented_batch, dim=0)
            return augmented_pcl, augmented_batch
        else:
            return augmented_pcl, batch_tensor
