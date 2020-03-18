import torch
from itertools import count


#from enum import Enum
#class DataForm():
#   IMAGE
#   BINARY_MAP
#   POINTS
#   RECTANGLES
#   POLYGONS
# TODO(anguelos) add dataset roles


class DeterministicImageAugmentation(object):
    """Deterministic augmentation functor and its factory.

    This class is the base class realising an augmentation.
    In order to create a new augmentation the forward method and the factory class-function have to be defined.
    If a constructor is defined it must call super().__init__(**kwargs).

    The **kwargs on the constructor define all needed variables for generating random augmentations.
    The factory method returns a lambda that instanciates augmentatations.
    Any parameter about the augmentations randomness should be passed by the factory to the constructor and used inside
    forward's definition.
    """
    _ids = count(0)
    # since instance seeds are the global seed plus the instance counter, starting from 1000000000 makes a collision
    # highly improbable. Although no guaranty of uniqueness of instance seed is made.
    _global_seed = torch.LongTensor(1).random_(1000000000, 2000000000).item()

    @staticmethod
    def reset_all_seeds(global_seed=None):
        SpatialImageAugmentation._ids = count(0)
        if global_seed is None:
            global_seed = 0
        SpatialImageAugmentation._global_seed = global_seed


    def __init__(self, **kwargs):
        # TODO (anguelos) maybe checking for id and seed is redundant?
        if "id" not in kwargs.keys():
            self.id = next(SpatialImageAugmentation._ids)
        else:
            self.id = kwargs["id"]
        if "seed" not in kwargs.keys():
            self.seed = self.id + SpatialImageAugmentation._global_seed
        else:
            self.seed = kwargs["seed"]
        self._params = kwargs
        self._params.update({"id":self.id,"seed":self.seed})
        self.__dict__.update(**kwargs)
        # TODO (anguelos) if this self.__dict__ hack is gone, DeterministicImageAugmentation can inherit from torch.nn.Module

    def __call__(self, tensor_image):
        print("Call happened")
        device = tensor_image.device
        with torch.random.fork_rng(devices=(device,)):
            torch.manual_seed(self.seed)
            return self.forward(tensor_image)

    def __repr__(self):
        # TODO (anguelos) make eval(repr(a)) a copy constructor. what about self.id and self.seed?
        parameters = "("+" ".join([",{}={}".format(k,repr(v)) for k,v in self._params.items()])+")"
        return self.__class__.__name__ + parameters

    def __eq__(self, obj):
        return self.__class__ is obj.__class__ and self.id == obj.id and self.seed == obj.seed

    # Methods for extending the class
    def forward(self, tensor_image):
        """Distorts a single image.

        Abstract method that must be implemented by all augmentations.

        :param tensor_image: Images are 3D tensors of [#channels, width, height] size.
        :return: A new 3D tensor [#channels, width, height] with the new image.
        """
        raise NotImplementedError()

    @classmethod
    def factory(cls):
        """Creates a lambda expression that is a factory of image augmentations.

        This function must be implemented by any child class.

        :return: A lambda that generates instances of deterministic image augmentors
        """
        raise NotImplementedError()

    @property
    def preserves_geometry(self):
        raise NotImplementedError()


class ChannelImageAugmentation(DeterministicImageAugmentation):
    @property
    def preserves_geometry(self):
        return True


class SpatialImageAugmentation(DeterministicImageAugmentation):
    @property
    def preserves_geometry(self):
        return False
