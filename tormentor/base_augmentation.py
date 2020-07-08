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

    def augment_image(self, image_tensor:torch.Tensor):
        device = image_tensor.device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            n_dims = len(image_tensor.size())
            if n_dims == 3:
                return self.forward_batch_img(image_tensor.unsqueeze(dim=0))[0, :, :, :]
            elif n_dims == 4:
                return self.forward_batch_img(image_tensor)
            else:
                raise ValueError("image_tensor must represent a sample or a batch")

    def augment_sampling_field(self, sf:SamplingField):
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

    def augment_pointcloud(self, pc: PointCloudOneOrMore, image_tensor:torch.FloatTensor):
        if isinstance(pc, list):
            device = pc[0][0].device
        else:
            device = pc[0].device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            if isinstance(pc, list):
                assert len(image_tensor.size()) == 4
                out_pc = self.forward_pointcloud(pc, image_tensor)
                return pc
            elif isinstance(pc[0], torch.Tensor):
                print(image_tensor.size())
                assert len(image_tensor.size()) == 3
                out_pc = self.forward_pointcloud([pc], image_tensor.unsqueeze(dim=0))[0]
                return out_pc
            else:
                raise ValueError("point clouds must represent a sample or a batch")

    def __call__(self, *args):
        # practiaclly method overloading
        # if I could only test objects for Typing Generics
        assert len(args) == 1 or len(args) == 2
        if len(args) == 2: # pointcloud and image tensor
            pointcloud, image_tensor = args
            return self.augment_pointcloud(pointcloud, image_tensor)
        elif isinstance(args[0], tuple): # sampling field
            assert (2 <= len(args[0][0].size()) <= 3) and len(args[1].size()) == 2
            return self.augment_sampling_field(args[0])
        elif isinstance(args[0], torch.Tensor): # image
            assert 3 <= len(args[0].size()) <= 4
            return self.augment_image(args[0])


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

    # def forward_sample_img(self, tensor_image):
    #     """Distorts a single image.
    #
    #     Abstract method that must be implemented by all augmentations.
    #
    #     :param tensor_image: Images are 3D tensors of [#channels, width, height] size.
    #     :return: A create_persistent 3D tensor [#channels, width, height] with the create_persistent image.
    #     """
    #     return self.forward_batch_img(tensor_image.unsqueeze(dim=0))[0, :, :, :]

    def forward_batch_img(self, batch_tensor):
        """Distorts a a batch of one or more images.

        :param batch_tensor: Images are 4D tensors of [batch_size, #channels, width, height] size.
        :return: A create_persistent 4D tensor [batch_size, #channels, width, height] with the create_persistent image.
        """
        #assert type(self).forward_batch_img is not DeterministicImageAugmentation.forward_sample_img
        augmented_tensors = []
        for n in range(batch_tensor.size(0)):
            augmented_tensors.append(self.forward_sample_img(batch_tensor[n, :, :, :]))
        return torch.cat(augmented_tensors, dim=0)

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


    def forward_bboxes(self, bboxes:torch.FloatTensor, image_tensor=None, width_height=None):
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
        raise NotImplementedError


    def forward_pointcloud(self, pc: PointCloud, batch_tensor: torch.FloatTensor)->PointCloud:
        """Applies a transformation on normalised coordinate points.

        This method assumes the transform differs by

        Args:
            pc (torch.Tensor, torch.Tensor): Normalised [-1, 1] coordinates to be transformed. the tensors first
                dimension is the batch dimension.
            batch_tensor (): A batch image tensor used to infer the size to populate the sampling field.

        Returns:

        """
        raise NotImplementedError()


    def forward_sampling_field(self, coords: SamplingField)->SamplingField:
        """Defines a spatial transform.

        Args:
            coords (): a tuple with two 3D float tensors each having a size of [Batch x Width x Height]. X and Y
                coordinates are in reference the range [-1, 1].

        Returns:
            The augmented samplint field.

        """
        raise NotImplementedError()


def aug_parameters(**kwargs):
    """Decorator that assigns random parameter ranges to augmentation and creates the augmentation's constructor.

    """
    default_params = kwargs

    def register_default_random_parameters(cls):
        setattr(cls, "default_params", default_params.copy())

        # Creating dynamically a constructor
        rnd_param = "\n\t".join((f"self.{k} = {k}" for k, v in default_params.items()))
        default_params["id"] = None
        default_params["seed"] = None
        param_str = ", ".join((f"{k}={repr(v)}" for k, v in default_params.items()))
        constructor_str = f"def __init__(self, {param_str}):\n\tDeterministicImageAugmentation.__init__(self, id=id, seed=seed)\n\t{rnd_param}"
        exec_locals = {"__class__": cls}
        exec(constructor_str, globals(), exec_locals)
        setattr(cls, "__init__", exec_locals["__init__"])
        return cls
    return register_default_random_parameters


class ChannelImageAugmentation(DeterministicImageAugmentation):
    @property
    def applicable_on_mask(self):
        return False

    def forward_mask(self,X):
        return X

    def forward_sampling_field(self, coords: SamplingField)->SamplingField:
        return coords

    def forward_bboxes(self, bboxes:torch.FloatTensor, image_tensor=None, width_height=None):
        return bboxes


    def forward_pointcloud(self, pc: PointCloud, batch_tensor: torch.FloatTensor)->PointCloud:
        return pc



class SpatialImageAugmentation(DeterministicImageAugmentation):
    #outside_field = 10

    @staticmethod
    def functional_sampling_field(self, coords: SamplingField)->SamplingField:
        raise NotImplementedError()

    def forward_batch_img(self, batch_tensor):
        batch_sz, channels, width, height = batch_tensor.size()
        print("BCWH Input Size:", batch_tensor.size())
        batch_tensor=batch_tensor.transpose(2,3) # BCWH -> BCHW
        print("BCHW Input Size:", batch_tensor.size())

        #xy_coords = K.utils.create_meshgrid(width, height, normalized_coordinates=True, device=batch_tensor.device)
        xy_coords = K.utils.create_meshgrid(height, width, normalized_coordinates=True, device=batch_tensor.device)
        xy_coords = xy_coords.repeat(batch_sz, 1, 1, 1)
        x_coords = xy_coords[:, :, :, 0]
        y_coords = xy_coords[:, :, :, 1]

        x_coords, y_coords = self.forward_sampling_field((x_coords, y_coords))

        # TODO (anguelos)
        xy_coords = torch.cat((x_coords.unsqueeze(dim=-1), y_coords.unsqueeze(dim=-1)), dim=3)
        #augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor.transpose(2, 3), xy_coords[:, :, :, [1, 0]])

        #augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, xy_coords[:, :, :, [1,0]].transpose(1,2))
        print("Before GS:",batch_tensor.size(), xy_coords.size())
        augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, xy_coords[:, :, :, [0,1]])
        print("After GS:",augmented_batch_tensor.size(), xy_coords.size())

        print("BCHW Return Size:", augmented_batch_tensor.size())
        augmented_batch_tensor=augmented_batch_tensor.transpose(2,3) # BCHW -> BCWH
        print("BCWH Return Size:", augmented_batch_tensor.size())
        return augmented_batch_tensor

    def forward_sampling_field(self, coords: SamplingField) -> SamplingField:
        state = self.generate_batch_state(coords)
        return type(self).functional_sampling_field(*((coords,) + state))

    def forward_bboxes(self, bboxes:torch.FloatTensor, image_tensor=None, width_height=None) -> torch.FloatTensor:
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
            width=image_tensor.size(-2)
            height=image_tensor.size(-1)
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

    def forward_pointcloud(self, pcl: PointCloudList, batch_tensor: torch.FloatTensor)->PointCloud:
        """Applies a transformation on normalised coordinate points.

        This method assumes the transform differs by

        Args:
            pc (torch.Tensor, torch.Tensor): Normalised [-1, 1] coordinates to be transformed. the tensors first
                dimension is the batch dimension.
            batch_tensor (): A batch image tensor used to infer the size to populate the sampling field.

        Returns:

        """
        batch_sz, _, width, height = batch_tensor.size()

        sampling_field_tensor = K.utils.create_meshgrid(width, height, normalized_coordinates=True, device=batch_tensor.device)
        sampling_field_tensor = sampling_field_tensor.repeat(batch_sz, 1, 1, 1)
        in_sampling_field = sampling_field_tensor[:, :, :, 0], sampling_field_tensor[:, :, :, 1]

        out_sampling_field = self.forward_sampling_field(in_sampling_field)
        out_pcl = []
        for batch_n in range(batch_sz):
            pc = pcl[batch_n]
            #pc_normalised = pc[0].view([-1])/(width*.5)-1, pc[1].view([-1])/(height*.5)-1
            pc_normalised = pc[0]/(width*.5)-1, pc[1]/(height*.5)-1
            X_dst, Y_dst = out_sampling_field[0][batch_n, :, :].view(-1), out_sampling_field[1][batch_n, :, :].view(-1)
            X_src, Y_src = in_sampling_field[0][batch_n, :, :].view(-1), in_sampling_field[1][batch_n, :, :].view(-1)
            nearest_neighbor_src_X = torch.empty_like(pc_normalised[0])
            nearest_neighbor_src_Y = torch.empty_like(pc_normalised[1])

            # TODO(anguelos) should we indirectly allow control over gpu
            step = 10000000 // (width*height) # This is about GPU memory

            for n in range(0, pc_normalised[0].size(0), step):
                pc_x, pc_y = pc_normalised[0][n: n+step], pc_normalised[1][n:n+step]
                #euclidean = ((X_src.view(1, -1)-pc_x.view(-1, 1))**2+(Y_src.view(1, -1)-pc_y.view(-1, 1))**2)
                euclidean = ((X_dst.view(1, -1)-pc_x.view(-1, 1))**2+(Y_dst.view(1, -1)-pc_y.view(-1, 1))**2)

                #plt.imshow(euclidean[-1, :].view([width, height]))
                #plt.show()

                idx = torch.argmin(euclidean, dim=1)
                #print(idx)
                #print("\n\nbefore nearest_neighbor_src_X\n",nearest_neighbor_src_X)
                nearest_neighbor_src_X[n:n+step] = X_src[idx][:]
                #print("\n\nafter nearest_neighbor_src_X\n",nearest_neighbor_src_X)
                nearest_neighbor_src_Y[n:n+step] = Y_src[idx]
                #print("X_src[idx]",X_src[idx])
            #plt.plot(pc_normalised[0], pc_normalised[1], ".")
            #plt.show()
            #plt.plot(nearest_neighbor_src_X, nearest_neighbor_src_Y,".")
            #plt.show()
            out_pc = (nearest_neighbor_src_X+1)*.5*width, (nearest_neighbor_src_Y+1) * .5 * height
            #print("out_pc[0]",out_pc[0])
            out_pcl.append(out_pc)
        #print(pcl)
        #print(out_pcl)
        return out_pcl








#sampling_field = -torch.clamp(sampling_field[0], -1, 1), -torch.clamp(sampling_field[1], -1, 1)

        # #Reversing the augmetation effect so that the augmentaion is rendered instead of sampled
        # #delta_sampling_field = out_sampling_field - in_sampling_field
        # #out_sampling_field = in_sampling_field - delta_sampling_field
        #
        # delta_sampling_field = out_sampling_field[0] - in_sampling_field[0], out_sampling_field[1] - in_sampling_field[1]
        # out_sampling_field = in_sampling_field[0] - delta_sampling_field[0], in_sampling_field[1] - delta_sampling_field[1]
        # del delta_sampling_field
        #
        # out_sampling_field = torch.clamp(out_sampling_field[0], -1, 1), torch.clamp(out_sampling_field[1], -1, 1)
        #
        # #sampling_field = -torch.clamp(sampling_field[0], -1, 1), -torch.clamp(sampling_field[1], -1, 1)
        #
        # #pixel_sampling_field = (sampling_field[0]+1) * .5 * height, (sampling_field[1]+1) * .5 * width
        # pixel_sampling_field = (in_sampling_field[0]+1) * .5 * width, (in_sampling_field[1]+1) * .5 * height
        #
        # #input_x, input_y = pc[1],pc[0]
        # input_x, input_y = pc
        # input_x = torch.clamp(input_x, 0, width - 1).contiguous().long()
        # input_y = torch.clamp(input_y, 0, height - 1).contiguous().long()
        #
        # input_sample_id = (torch.ones_like(pc[0]).cumsum(dim=0) - 1).contiguous().long()
        #
        #
        # output_x = pixel_sampling_field[0][input_sample_id.view(-1), input_x.view(-1), input_y.view(-1)].contiguous()
        # output_y = pixel_sampling_field[1][input_sample_id.view(-1), input_x.view(-1), input_y.view(-1)].contiguous()
        #
        # return output_y.view(input_x.size()), output_x.view(input_y.size())
        # #return output_x.view(input_x.size()), output_y.view(input_y.size())


    # @staticmethod
    # def render_functional_forward_points(batch_tensor, augmentation_type, state):
    #     """Facilitates pluging custom values
    #
    #     Args:
    #         batch_tensor (): The input batch tensor.
    #         augmentation_type (): The specific SpatialImageAugmentation to use
    #         state ():
    #
    #     Returns:
    #         An augmented batch tensor
    #
    #     """
    #     batch_sz, channels, width, height = batch_tensor.size()
    #     xy_coords = K.utils.create_meshgrid(width, height, normalized_coordinates=True, device=batch_tensor.device)
    #     xy_coords = xy_coords.repeat(batch_sz, 1, 1, 1)
    #     x_coords = xy_coords[:, :, :, 0]
    #     y_coords = xy_coords[:, :, :, 1]
    #     x_coords, y_coords = augmentation_type.functional_sampling_field(*(((x_coords, y_coords),) + state))
    #     xy_coords = torch.cat((x_coords.view(batch_sz, width, height, 1), y_coords.view(batch_sz, width, height, 1)), 3)
    #     augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, xy_coords)
    #     return augmented_batch_tensor










class AugmentationCascade(DeterministicImageAugmentation):
    @classmethod
    def create(cls, augmentation_list):

        ridx = cls.__qualname__.rfind("_")
        if ridx == -1:
            cls_oldname = cls.__qualname__
        else:
            cls_oldname = cls.__qualname__[:ridx]
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), augmentation_list)
        return new_cls

    def forward_sampling_field(self, coords: SamplingField):
        for augmentation_type in type(self).augmentation_list:
            augmentation = augmentation_type()
            coords = augmentation(coords)
        return coords

    def forward_batch_img(self, batch_tensor):
        for augmentation_type in type(self).augmentation_list:
            augmentation = augmentation_type()
            batch_tensor = augmentation.forward_batch_img(batch_tensor)
        return batch_tensor


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
        new_cls_name = f"{cls_oldname}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), new_parameters)
        return new_cls

    def forward_sampling_field(self, coords: SamplingField):
        batch_sz = coords[0].size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch_x = []
        augmented_batch_y = []
        for sample_n in range(batch_sz):
            sample_coords = sample_coords[0][sample_n:sample_n + 1, :, :], sample_coords[1][sample_n:sample_n + 1, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            sample_x, sample_y = augmentation(sample_coords)
            augmented_batch_x.append(sample_x)
            augmented_batch_y.append(sample_y)
        augmented_batch_x = torch.cat(augmented_batch_x, dim=0)
        augmented_batch_y = torch.cat(augmented_batch_y, dim=0)
        return augmented_batch_x, augmented_batch_y


    def forward_batch_img(self, batch_tensor):
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmentation = type(self).available_augmentations[augmentation_ids[sample_n]]()
            augmented_sample = augmentation.forward_batch_img(sample_tensor)
            augmented_batch.append(augmented_sample)
        augmented_batch = torch.cat(augmented_batch, dim=0)
        return augmented_batch
