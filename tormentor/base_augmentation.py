import torch
from itertools import count
import types
import kornia as K
from .random import Distribution, Uniform, Bernoulli, Categorical
from typing import Tuple, Union, List
from matplotlib import pyplot as plt


SamplingField = Tuple[torch.FloatTensor, torch.FloatTensor]
PointCloud = Tuple[torch.FloatTensor, torch.FloatTensor]
PointCloudList = List[PointCloud]
PointCloudsImages = Tuple[PointCloudList, torch.FloatTensor]
PointCloudOneOrMore = Union[PointCloudList, PointCloud]
SpatialAugmentationState = Tuple[torch.Tensor, ...]


class AugmentationLayer(torch.nn.Module):
    """Generates deterministic augmentations.

    It is a serializable generator that can override default random get_distribution_parameters.
    """
    def __init__(self, augmentation_cls, **kwargs):
        super().__init__()
        assert set(kwargs.keys()) <= set(augmentation_cls.distributions.keys())
        self.augmentation_distributions = {k: v.copy() for k, v in augmentation_cls.distributions.items()}
        overridden_distributions = {k: v for k, v in kwargs.items() if k in augmentation_cls.distributions.keys()}
        self.augmentation_distributions.update(overridden_distributions)
        self.augmentation_distributions = torch.nn.ModuleDict(self.augmentation_distributions)
        self.augmentation_cls = augmentation_cls

    def forward(self, x, is_mask=False, is_points=False):
        if self.training:
            return self.create_persistent()(x, is_mask=is_mask, is_points=is_points)
        else:
            return x

    def create_persistent(self):
        return self.augmentation_cls()


class DeterministicImageAugmentation(object):
    """Deterministic augmentation functor and its factory.

    This class is the base class realising an augmentation.
    In order to create a create_persistent augmentation the forward_sample_img method and the factory class-function have to be defined.
    If a constructor is defined it must call super().__init__(**kwargs).

    The **kwargs on the constructor define all needed variables for generating random augmentations.
    The factory method returns a lambda that instanciates augmentatations.
    Any parameter about the augmentations randomness should be passed by the factory to the constructor and used inside
    forward_sample_img's definition.
    """

    _ids = count(0)
    # since instance seeds are the global seed plus the instance counter, starting from 1000000000 makes a collision
    # highly improbable. Although no guaranty of uniqueness of instance seed is made.
    # in case of a multiprocess parallelism, this minimizes chances of collision
    _global_seed = torch.LongTensor(1).random_(1000000000, 2000000000).item()
    #distributions = {"occurence": Bernoulli(prob=.3)}
    #occurence = Bernoulli(prob=1.)
    distributions = {}

    @staticmethod
    def reset_all_seeds(global_seed=None):
        DeterministicImageAugmentation._ids = count(0)
        if global_seed is None:
            global_seed = 0
        DeterministicImageAugmentation._global_seed = global_seed


    def __init__(self, id=None, seed=None):
        if id is None:
            self.id = next(DeterministicImageAugmentation._ids)
        else:
            self.id = id
        if seed is None:
            self.seed = self.id + DeterministicImageAugmentation._global_seed
        else:
            self.seed = seed

    def like_me(self):
        """Returns a new instance of the same class as self."""
        return type(self)()

    def augment_image(self, image_tensor: torch.FloatTensor):
        device = image_tensor.device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            n_dims = len(image_tensor.size())
            if n_dims == 3:
                return self.forward_img(image_tensor.unsqueeze(dim=0))[0, :, :, :]
            elif n_dims == 4:
                return self.forward_img(image_tensor)
            else:
                raise ValueError("image_tensor must represent a sample or a batch")

    def augment_mask(self, image_tensor: torch.Tensor):
        device = image_tensor.device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            n_dims = len(image_tensor.size())
            if n_dims == 3:
                print("Base augment_mask Start",image_tensor.sum())
                res = self.forward_mask(image_tensor.unsqueeze(dim=0))[0, :, :]
                print("Base augment_mask End",image_tensor.sum())
                return res
            elif n_dims == 4:
                return self.forward_mask(image_tensor)
            else:
                raise ValueError("mask_tensor must represent a sample [HxW] or a batch [BxHxW]")

    def augment_sampling_field(self, sf: SamplingField):
        assert sf[0].size() == sf[1].size()
        device = sf[0].device
        with torch.random.fork_rng(devices=(device,)):
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

    def augment_pointcloud(self, pc: PointCloudOneOrMore, image_tensor: torch.FloatTensor, compute_img:bool):
        if isinstance(pc, list):
            device = pc[0][0].device
        else:
            device = pc[0].device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            if isinstance(pc, list):
                assert len(image_tensor.size()) == 4
                out_pc, out_img = self.forward_pointcloud(pc, image_tensor, compute_img=compute_img)
                return out_pc, out_img
            elif isinstance(pc[0], torch.Tensor):
                assert len(image_tensor.size()) == 3
                out_pc, out_img = self.forward_pointcloud([pc], image_tensor.unsqueeze(dim=0), compute_img=compute_img)
                out_pc = out_pc[0]
                out_img = out_img[0, :, :, :]
                return out_pc, out_img
            else:
                raise ValueError("point clouds must represent a sample or a batch")

    def __call__(self, *args, compute_img: bool = True, is_mask: bool = False):
        """Run augmentation on data of varius types

        Args:
            *args:
            compute_img:
            is_mask: If args is a single Tensor, when this is True it will be treated as a mask and when false as an
                image. When args is not a single tensor this will be ignored.

        Returns:

        """
        print("__call__")
        # practiaclly method overloading
        # if I could only test objects for Typing Generics
        assert len(args) == 1 or len(args) == 2
        if len(args) == 2: # pointcloud and image tensor
            print("__call__:pc")
            pointcloud, image_tensor = args
            return self.augment_pointcloud(pointcloud, image_tensor, compute_img)
        elif isinstance(args[0], tuple): # sampling field
            print("__call__:sf")
            assert (2 <= len(args[0][0].size()) <= 3) and len(args[1].size()) == 2
            return self.augment_sampling_field(args[0])
        elif isinstance(args[0], torch.Tensor) and not is_mask: # image
            print("__call__:img")
            assert 3 <= len(args[0].size()) <= 4
            return self.augment_image(args[0])
        elif isinstance(args[0], torch.Tensor) and is_mask:
            print("__call__:mask")
            assert 3 <= len(args[0].size()) <= 4
            return self.augment_mask(args[0])
        else:
            print(args[0].dtype)
            raise ValueError


    def __repr__(self):
        param_names = ("id", "seed")
        param_assignments = ", ".join(["{}={}".format(k, repr(self.__dict__[k])) for k in param_names])
        return self.__class__.__qualname__ + "(" + param_assignments + ")"

    def __eq__(self, obj):
        return self.__class__ is obj.__class__ and self.id == obj.id and self.seed == obj.seed

    def __str__(self):
        functions = (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType)
        attribute_strings = []
        for name in self.__dict__.keys():
            attribute = getattr(self, name)
            if (not isinstance(attribute, functions)) and (not name.startswith("__")):
                attribute_strings.append(f"{name}:{repr(attribute)}")
        return self.__class__.__qualname__ + ":\n\t" + "\n\t".join(attribute_strings)

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> SpatialAugmentationState:
        """Generates deterministic state for each augmentation.

        Returns: a tuple of tensors representing the complete state of the augmentaion so that a

        """
        raise NotImplementedError()

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        """Distorts a a batch of one or more images.

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
        # if batch_tensor.dtype is torch.long:
        #     # assuming this has to be a dense segmentation to be processed with onehot encoding.
        #     assert len(batch_tensor.size()) ==3 # no channels in pixel labels
        #     n_classes = batch_tensor.max() + 1
        #     batch_size, height, width = batch_tensor.size()
        #     batch_tensor = batch_tensor.unsqueeze(dim=1)
        #     batch_onehot = torch.FloatTensor(batch_size, n_classes, height, width)
        #     batch_onehot.zero_()
        #     ones = torch.ones(batch_tensor.size())
        #     batch_onehot.scatter_(1, batch_tensor, ones)
        #     augmented_batch_onehot = self.forward_batch_img(batch_onehot)
        #     augmented_batch_onehot[:, 0, :, :] += .000000001 # adding a tie braker
        #     return augmented_batch_onehot.argmax(dim=1)
        # else:
        #     return self.forward_batch_img(batch_tensor)



    @classmethod
    def get_distributions(cls):
        return {k: v.copy() for k, v in cls.__dict__.items() if isinstance(v, Distribution)}

    @classmethod
    def override_distributions(cls, requires_grad=False, **kwargs):
        cls_distributions = cls.get_distributions()
        assert all([isinstance(v, Distribution) for v in kwargs.values()])
        assert set(kwargs.keys()) <= set(cls_distributions.keys())
        cls_distributions.update(kwargs)
        for cls_distribution in cls_distributions.values():
            for parameter in cls_distribution.get_distribution_parameters().values():
                parameter.requires_grad_(requires_grad)
        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), cls_distributions)
        return new_cls

    @property
    def device(self):
        type(self).occurence.prob.device


    def forward_bboxes(self, bboxes:torch.FloatTensor, image_tensor=None, width_height=None) -> torch.FloatTensor:
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
            width=image_tensor.size(-1)
            height=image_tensor.size(-2)
        else:
            width, height = width_height
        normalise_bboxes = torch.tensor([[width * .5, height * .5, width * .5, height * .5]])
        normalised_bboxes = bboxes/normalise_bboxes -1
        bboxes_left = normalised_bboxes[:, 0].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_top = normalised_bboxes[:, 1].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_right = normalised_bboxes[:, 2].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_bottom = normalised_bboxes[:, 3].unsqueeze(dim=0).unsqueeze(dim=0)
        bboxes_x = torch.concat([bboxes_left, bboxes_right, bboxes_right, bboxes_left], dim=1)
        bboxes_y = torch.concat([bboxes_top, bboxes_top, bboxes_bottom, bboxes_bottom], dim=1)

        pointcloud = (bboxes_x, bboxes_y)
        pointcloud = self.forward_pointcloud(pointcloud)
        pointcloud = torch.clamp(pointcloud[0], -1, 1), torch.clamp(pointcloud[0], -1, 1)

        left = ((pointcloud[0].min(dim=1) + 1) * .5 * width).view(-[1,1])
        right = ((pointcloud[0].max(dim=1) + 1) * .5 * width).view(-[1,1])
        top = ((pointcloud[1].min(dim=1)+1) *.5 * height).view(-[1,1])
        bottom = ((pointcloud[1].max(dim=1) + 1) * .5 * height).view(-[1,1])
        result_bboxes = torch.concat([left, top, right, bottom], dim=1)
        return result_bboxes


    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor, compute_img:bool) -> PointCloudsImages:
        """Applies a transformation on normalised coordinate points.

        This method assumes the transform differs by

        Args:
            pc (torch.Tensor, torch.Tensor): Normalised [-1, 1] coordinates to be transformed. the tensors first
                dimension is the batch dimension.
            batch_tensor (): A batch image tensor used to infer the size to populate the sampling field.

        Returns:

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
            pc_normalised = pc[0]/(width*.5)-1, pc[1]/(height*.5)-1
            X_dst, Y_dst = out_sampling_field[0][batch_n, :, :].view(-1), out_sampling_field[1][batch_n, :, :].view(-1)
            X_src, Y_src = in_sampling_field[0][batch_n, :, :].view(-1), in_sampling_field[1][batch_n, :, :].view(-1)
            nearest_neighbor_src_X = torch.empty_like(pc_normalised[0])
            nearest_neighbor_src_Y = torch.empty_like(pc_normalised[1])
            # TODO(anguelos) should we indirectly allow control over gpu
            step = 10000000 // (width*height) # This is about GPU memory
            for n in range(0, pc_normalised[0].size(0), step):
                pc_x, pc_y = pc_normalised[0][n: n+step], pc_normalised[1][n:n+step]
                euclidean = ((X_dst.view(1, -1)-pc_x.view(-1, 1))**2+(Y_dst.view(1, -1)-pc_y.view(-1, 1))**2)
                idx = torch.argmin(euclidean, dim=1)
                nearest_neighbor_src_X[n:n+step] = X_src[idx][:]
                nearest_neighbor_src_Y[n:n+step] = Y_src[idx]
            out_pc = (nearest_neighbor_src_X+1)*.5*width, (nearest_neighbor_src_Y+1) * .5 * height
            out_pcl.append(out_pc)
        if compute_img:
            out_coords = torch.cat((out_sampling_field[0].unsqueeze(dim=-1), out_sampling_field[1].unsqueeze(dim=-1)), dim=3)
            augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, out_coords, align_corners=True)
            return out_pcl, augmented_batch_tensor
        else:
            return out_pcl, batch_tensor

    def forward_sampling_field(self, coords: SamplingField)->SamplingField:
        """Defines a spatial transform.

        Args:
            coords (): a tuple with two 3D float tensors each having a size of [Batch x Height x Width]. X and Y
                coordinates are in reference the range [-1, 1].

        Returns:
            The augmented samplint field.

        """
        raise NotImplementedError()


# def aug_parameters(**kwargs):
#     """Decorator that assigns random parameter ranges to augmentation and creates the augmentation's constructor.
#
#     """
#     default_params = kwargs
#
#     def register_default_random_parameters(cls):
#         setattr(cls, "default_params", default_params.copy())
#
#         # Creating dynamically a constructor
#         rnd_param = "\n\t".join((f"self.{k} = {k}" for k, v in default_params.items()))
#         default_params["id"] = None
#         default_params["seed"] = None
#         param_str = ", ".join((f"{k}={repr(v)}" for k, v in default_params.items()))
#         constructor_str = f"def __init__(self, {param_str}):\n\tDeterministicImageAugmentation.__init__(self, id=id, seed=seed)\n\t{rnd_param}"
#         exec_locals = {"__class__": cls}
#         exec(constructor_str, globals(), exec_locals)
#         setattr(cls, "__init__", exec_locals["__init__"])
#         return cls
#     return register_default_random_parameters


class StaticImageAugmentation(DeterministicImageAugmentation):
    """Parent class for augmentations that don't move things around.

    All classes that do descend from this are expected to be neutral for pointclouds, sampling fields although they
    might be erasing regions of the images so they are not gurantied to be neutral to masks.
    Every class were image pixels move around rather that just stay static, should be a descendant of
    SpatialImageAugmentation.
    """
    @classmethod
    def functional_image(cls, image_tensor, *state):
        raise NotImplementedError()

    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        print("StaticImageAugmentation.forward_mask")
        return X

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        return coords

    def forward_bboxes(self, bboxes:torch.FloatTensor, image_tensor=None, width_height=None):
        return bboxes

    def forward_pointcloud(self, pc: PointCloud, batch_tensor: torch.FloatTensor, compute_img:bool)->PointCloud:
        print("StaticImageAugmentation.forward_pointcloud",compute_img)
        if compute_img:
            return pc, self.forward_img(batch_tensor)
        else:
            return pc, batch_tensor

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        print("StaticImageAugmentation.forward_img")
        state = self.generate_batch_state(batch_tensor)
        return type(self).functional_image(*((batch_tensor,) + state))


class SpatialImageAugmentation(DeterministicImageAugmentation):
    """Parent class for augmentations that move things around.

    Every class were image pixels move around rather that just change should be a descendant of this class.
    All classes that do not descend from this class are expected to be neutral for pointclouds, and sampling fields and
    should be descendants of StaticImageAugmentation.
    """
    @classmethod
    def functional_sampling_field(cls, coords: SamplingField, *state) -> SamplingField:
        raise NotImplementedError()

    def forward_img(self, batch_tensor):
        batch_sz, channels, height, width = batch_tensor.size()

        xy_coords = K.utils.create_meshgrid(height, width, normalized_coordinates=True, device=batch_tensor.device)
        xy_coords = xy_coords.repeat(batch_sz, 1, 1, 1)
        x_coords = xy_coords[:, :, :, 0]
        y_coords = xy_coords[:, :, :, 1]

        x_coords, y_coords = self.forward_sampling_field((x_coords, y_coords))

        # TODO (anguelos)
        xy_coords = torch.cat((x_coords.unsqueeze(dim=-1), y_coords.unsqueeze(dim=-1)), dim=3)
        augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, xy_coords, align_corners=True)
        return augmented_batch_tensor

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        state = self.generate_batch_state(coords)
        return type(self).functional_sampling_field(*((coords,) + state))

    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_img(X)


class AugmentationCascade(DeterministicImageAugmentation):
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

    def augment_sampling_field(self, sf: SamplingField)->SamplingField:
        device = sf[0].device
        with torch.random.fork_rng(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                sf = augmentation.forward_sampling_field(sf)
        return sf

    def augment_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = image_tensor.device
        with torch.random.fork_rng(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                image_tensor = augmentation.forward_img(image_tensor)
        return image_tensor

    def augment_mask(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = image_tensor.device
        with torch.random.fork_rng(devices=(device,)):
            for augmentation in self.augmentations:
                torch.manual_seed(augmentation.seed)
                image_tensor = augmentation.forward_mask(image_tensor)
        return image_tensor

    def forward_sampling_field(self, coords: SamplingField) ->SamplingField:
        return NotImplemented # determinism forbids running under other seed

    def forward_bboxes(self, bboxes:torch.FloatTensor, image_tensor=None, width_height=None) -> torch.FloatTensor:
        return NotImplemented # determinism forbids running under other seed

    def forward_img(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        return NotImplemented # determinism forbids running under other seed

    def forward_mask(self, batch_tensor: torch.LongTensor) -> torch.LongTensor:
        return NotImplemented # determinism forbids running under other seed

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor, compute_img:bool) -> PointCloudsImages:
        return NotImplemented # determinism forbids running under other seed

    @classmethod
    def create(cls, augmentation_list):
        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000, 9000000, (1,)).item()}"
        new_cls = type(new_cls_name, (cls,), {"augmentation_list": augmentation_list, "aumentation_instance_list":[aug() for aug in augmentation_list]})
        return new_cls

    @classmethod
    def get_distributions(cls):
        res = {}
        for n, contained_augmentation in enumerate(cls.augmentation_list):
            res.update({(n, k): v.copy() for k, v in contained_augmentation.get_distributions()})
        return res


class AugmentationChoice(DeterministicImageAugmentation):
    @classmethod
    def create(cls, augmentation_list, requires_grad=False):
        new_parameters = {"choice": Categorical(len(augmentation_list)),"available_augmentations": augmentation_list}
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
