import torch
from itertools import count
import types
import kornia as K
from .random import Distribution, Uniform, Bernoulli, Categorical
from typing import Tuple, Union


SamplingField = Tuple[torch.FloatTensor, torch.FloatTensor]
SpatialAugmentationState = Tuple[torch.Tensor, ...]


class BatchAugmentationFactory(torch.nn.Module):
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


class BatchMultiAugmentationFactory(BatchAugmentationFactory):
    def __init__(self, augmentation_cls_list):
        super().__init__()
        self.augmentation_distributions = {}
        self.augmentation_cls_list = augmentation_cls_list
        for cls in augmentation_cls_list:
            cls_distributions = {f"{cls.__qualname__}.{k}": v.copy() for k, v in cls.distributions.items()}
            self.augmentation_distributions.update(cls_distributions)
        self.augmentation_distributions = torch.nn.ModuleDict(self.augmentation_distributions)
        self.selection_distribution = Categorical(n_categories=len(augmentation_cls_list))

    def forward(self, x, is_mask=False, is_points=False):
        if self.training:
            return self.create_persistent()(x, is_mask=is_mask, is_points=False)
        else:
            return x

    def create_persistent(self):
        self.selection_distribution.sample(0).item()
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
    occurence = Bernoulli(prob=1.)
    distributions = {}

    @staticmethod
    def reset_all_seeds(global_seed=None):
        SpatialImageAugmentation._ids = count(0)
        if global_seed is None:
            global_seed = 0
        SpatialImageAugmentation._global_seed = global_seed


    def __init__(self, id=None, seed=None):
        if id is None:
            self.id = next(DeterministicImageAugmentation._ids)
        else:
            self.id = id
        if seed is None:
            self.seed = self.id + DeterministicImageAugmentation._global_seed
        else:
            self.seed = seed

    def __call__(self, tensor_image, is_mask=False):
        #if not (self.applicable_on_mask or is_mask):
        #    return tensor_image
        device = tensor_image.device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            if len(tensor_image.size()) == 3:
                if type(self).occurence():
                    return self.forward_sample_img(tensor_image)
                else:
                    return tensor_image
            elif len(tensor_image.size()) == 4:
                # faster but not differentiable
                #apply_on = self.occurence(tensor_image.size(0)).byte()
                #applied = self.forward_batch_img(tensor_image[apply_on, :, :, :])
                #results = tensor_image.clone()
                #results[apply_on, :, :, :] = applied
                #return results
                #slower but differentiable
                applied = self.forward_batch_img(tensor_image)
                #mask = self.occurence(tensor_image.size(0)).unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
                #return applied * mask + tensor_image * (1 - mask)
                return applied

            else:
                raise ValueError("Augmented tensors must be samples (3D tensors) or batches (4D tensors)")

    def __repr__(self):
        #param_names = [f"{k}" for k in self.distributions.keys()]
        param_names = ("id", "seed") #+ tuple(param_names)
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

    def forward_sample_img(self, tensor_image):
        """Distorts a single image.

        Abstract method that must be implemented by all augmentations.

        :param tensor_image: Images are 3D tensors of [#channels, width, height] size.
        :return: A create_persistent 3D tensor [#channels, width, height] with the create_persistent image.
        """
        return self.forward_batch_img(tensor_image.unsqueeze(dim=0))[0, :, :, :]

    def forward_batch_img(self, batch_tensor):
        """Distorts a a batch of one or more images.

        :param batch_tensor: Images are 4D tensors of [batch_size, #channels, width, height] size.
        :return: A create_persistent 4D tensor [batch_size, #channels, width, height] with the create_persistent image.
        """
        assert type(self).forward_batch_img is not DeterministicImageAugmentation.forward_sample_img
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
        #for k, v in kwargs.items():
        #    print(k, (type(v)))
        print(kwargs.values())
        assert all([isinstance(v, Distribution) for v in kwargs.values()])
        print(kwargs.keys())
        print(cls_distributions.keys())
        assert set(kwargs.keys()) <= set(cls_distributions.keys())
        cls_distributions.update(kwargs)
        print(cls_distributions)
        for cls_distribution in cls_distributions.values():
            for parameter in cls_distribution.get_distribution_parameters().values():
                parameter.requires_grad_(requires_grad)
        new_cls_name = f"{cls.__qualname__}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), cls_distributions)
        return new_cls

    @classmethod
    def factory(cls, **kwargs):
        return BatchAugmentationFactory(cls, **kwargs)

    @property
    def device(self):
        type(self).occurence.prob.device

    def forward_points(self, coords: SamplingField)->SamplingField:
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

    def forward_points(self, coords: SamplingField)->SamplingField:
        """Identity spatial augmentation.

        Args:
            coords ():

        Returns:

        """
        return coords


class SpatialImageAugmentation(DeterministicImageAugmentation):
    outside_field = 10

    @property
    def applicable_on_mask(self):
        return True

    def forward_mask(self, X):
        return self.forward(X)

    def forward_batch_img(self, batch_tensor):
        batch_sz, channels, width, height = batch_tensor.size()
        xy_coords = K.utils.create_meshgrid(width, height, normalized_coordinates=True, device=batch_tensor.device)
        xy_coords = xy_coords.repeat(batch_sz, 1, 1, 1)
        x_coords = xy_coords[:, :, :, 0]
        y_coords = xy_coords[:, :, :, 1]
        x_coords, y_coords = self.forward_points((x_coords, y_coords))
        xy_coords = torch.cat((x_coords.view(batch_sz, width, height, 1), y_coords.view(batch_sz, width, height, 1)), 3)
        augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, xy_coords)
        return augmented_batch_tensor

    def forward_points(self, coords: SamplingField)->SamplingField:
        state = self.generate_batch_state(coords)
        return type(self).functional_points(*((coords,) + state))

    @staticmethod
    def render_functional_forward_points(batch_tensor, augmentation_type, state):
        """Facilitates pluging custome values

        Args:
            batch_tensor (): The input batch tensor.
            augmentation_type (): The specific SpatialImageAugmentation to use
            state ():

        Returns:
            An augmented batch tensor

        """
        batch_sz, channels, width, height = batch_tensor.size()
        xy_coords = K.utils.create_meshgrid(width, height, normalized_coordinates=True, device=batch_tensor.device)
        xy_coords = xy_coords.repeat(batch_sz, 1, 1, 1)
        x_coords = xy_coords[:, :, :, 0]
        y_coords = xy_coords[:, :, :, 1]
        x_coords, y_coords = augmentation_type.functional_points(*(((x_coords, y_coords),)+state))
        xy_coords = torch.cat((x_coords.view(batch_sz, width, height, 1), y_coords.view(batch_sz, width, height, 1)), 3)
        augmented_batch_tensor = torch.nn.functional.grid_sample(batch_tensor, xy_coords)
        return augmented_batch_tensor


class AugmentationChoice(DeterministicImageAugmentation):
    @classmethod
    def create(cls, augmentation_list, requires_grad=False):
        all_distributions = {"choice": Categorical(len(augmentation_list), requires_grad=requires_grad)}
        for augmentation in augmentation_list:
            class_name = str(augmentation).split(".")[-1][:-2]
            cls_distributions = augmentation.get_distributions()
            cls_distributions = {f"{class_name}_{k}": v for k, v in cls_distributions.get_items()}
            all_distributions.update(cls_distributions)
        for cls_distribution in cls_distributions:
            for parameter in cls_distribution.get_distribution_parameters():
                parameter.requires_grad_(requires_grad)
        new_cls_name = f"{cls.__qualname__}_{torch.randint(1000000,9000000,(1,)).item()}"
        new_cls = type(new_cls_name, (cls,), cls_distributions)
        return new_cls

    def forward_points(self, coords: SamplingField):
        batch_sz = coords[0].size(0)
        augmentation_ids = type(self).choice(batch_sz)
        batch_augmented_x = []
        batch_augmented_y = []
        for sample_n in range(batch_sz):
            sample_coords = (c[sample_n:sample_n + 1, :, :] for c in coords)
            sample_augmented_x, sample_augmented_y = augmentation_ids[sample_n].forward_points(sample_coords)
            batch_augmented_x.append(sample_augmented_x)
            batch_augmented_y.append(sample_augmented_y)
        batch_augmented_x = torch.cat(batch_augmented_x, dim=0)
        batch_augmented_y = torch.cat(batch_augmented_y, dim=0)
        return batch_augmented_x, batch_augmented_y

    def forward_batch_img(self, batch_tensor):
        batch_sz = batch_tensor.size(0)
        augmentation_ids = type(self).choice(batch_sz)
        augmented_batch = []
        for sample_n in range(batch_sz):
            sample_tensor = batch_tensor[sample_n:sample_n + 1, :, :, :]
            augmented_sample = augmentation_ids[sample_n].forward_batch_img(sample_tensor)
            augmented_batch.append(augmented_sample)
        augmented_batch = torch.cat(augmented_batch, dim=0)
        return augmented_batch
