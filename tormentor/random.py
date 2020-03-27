from typing import Tuple, Union

import torch

TupleRange = Tuple[float, float]
Tuple2DRange = Union[Tuple[float, float], Tuple[float, float, float, float]]  # min max, min_x, max_x, min_y max_y
TensorSize = Union[torch.Size, int, Tuple[int], Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]
ImageSize = Union[Tuple[int, int], Tuple[torch.LongTensor, torch.LongTensor]]


class Distribution(object):
    def __init__(self, do_rsample):
        self.do_rsample = do_rsample

    def __call__(self, size: TensorSize = 1):
        raise NotImplementedError()

    def copy(self, do_rsample=None):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()


class Distribution2D(Distribution):
    def __init__(self, do_rsample):
        super().__init__(do_rsample)

    def get_rect_sizes(self, size: TensorSize = 1, image_total_size: ImageSize = (224, 224)) -> ImageSize:
        raise NotImplementedError()

    def get_rect_locations(self, rect_sizes: ImageSize, size: TensorSize = 1,
                           image_total_size: ImageSize = (224, 224)) -> ImageSize:
        raise NotImplementedError()

    def __call__(self, size: TensorSize = 1):
        raise NotImplementedError()

    def copy(self, do_rsample):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

class Uniform(Distribution):
    def __init__(self, value_range: TupleRange = (0.0, 1.0), do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.min = torch.autograd.Variable(torch.Tensor([value_range[0]]))
        self.max = torch.autograd.Variable(torch.Tensor([value_range[1]]))
        self.distribution = torch.distributions.Uniform(low=self.min, high=self.max)
        self.do_rsample = do_rsample

    def __repr__(self):
        range_str = f"({self.distribution.low.item()}, {self.distribution.high.item()})"
        param_str = f" do_rsample={self.do_rsample}"
        return f"{self.__class__.__qualname__}(value_range={range_str}, {param_str})"

    def __call__(self, size: TensorSize = 1) -> torch.Tensor:
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.distribution.sample(size).view(size)

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Uniform(value_range=(self.min.item(), self.max.item()), do_rsample=do_rsample)

    def parameters(self):
        return {"min": self.min, "max": self.max}


class Uniform2D(Distribution2D):
    def __init__(self, location: Tuple2DRange = (0.0, 1.0), do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        if len(location) == 2:
            self.horiz_min = self.vert_min = torch.autograd.Variable(torch.Tensor([location[0]]))
            self.horiz_max = self.vert_max = torch.autograd.Variable(torch.Tensor([location[1]]))
            self.horiz_distribution = torch.distributions.Uniform(low=self.horiz_min, high=self.horiz_max)
            self.vert_distribution = self.horiz_distribution
        else:  # len(size)==4
            self.horiz_min = torch.autograd.Variable(torch.Tensor([location[0]]))
            self.horiz_max = torch.autograd.Variable(torch.Tensor([location[1]]))
            self.vert_min = torch.autograd.Variable(torch.Tensor([location[2]]))
            self.vert_max = torch.autograd.Variable(torch.Tensor([location[3]]))
            self.horiz_distribution = torch.distributions.Uniform(low=self.horiz_min, high=self.horiz_max)
            self.vert_distribution = torch.distributions.Uniform(low=self.vert_min, high=self.vert_max)

    def __repr__(self) -> str:
        if self.vert_distribution is self.horiz_distribution:
            location_str = f"{self.__class__.__qualname__}(location=({self.horiz_min.item()}, {self.horiz_max.item()}))"
        else:
            horiz_str = f"{self.horiz_min.item()}, {self.horiz_max.item()}"
            vert_str = f"{self.vert_min.item()}, {self.vert_max.item()}"
            location_str = f"location=({horiz_str}, {vert_str})"
        return f"{self.__class__.__qualname__}({location_str}, do_rsample={self.do_rsample})"

    def __call__(self, size: TensorSize = 1):
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.horiz_distribution.sample(size).view(size), self.vert_distribution.sample(size).view(size)

    def get_rect_sizes(self, size: TensorSize = 1, image_total_size: ImageSize = (224, 224)) -> ImageSize:
        assert self.horiz_max.item() <= 1.0 and self.vert_max.item() <= 1.0 and self.horiz_min.item() >= 0.0 and self.vert_min.item() >= 0.0
        width, height = image_total_size
        x, y = self.__call__(size)
        x = (x * width).round().long64()
        y = (y * height).round().long64()
        return x, y

    def get_rect_locations(self, rect_sizes: ImageSize, size: TensorSize = 1, image_total_size: ImageSize = (224, 224)) -> ImageSize:
        assert self.horiz_max.item() <= 1.0 and self.vert_max.item() <= 1.0 and self.horiz_min.item() >= 0.0 and self.vert_min.item() >= 0.0
        width, height = image_total_size
        width = width - rect_sizes[0]
        height = height - rect_sizes[1]
        left, top = self.__call__(size)
        left = (left * width).round().long64()
        top = (top * height).round().long64()
        return left, top

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        if self.horiz_distribution is self.vert_distribution:
            value_range = (self.horiz_distribution.min.item(), self.horiz_distribution.max.item())
        else:
            horiz_range = (self.horiz_distribution.min.item(), self.horiz_distribution.max.item())
            vert_range = (self.vert_distribution.min.item(), self.vert_distribution.max.item())
            value_range = horiz_range + vert_range
        return Uniform(value_range=value_range, do_rsample=do_rsample)

    def parameters(self):
        if self.horiz_distribution is self.vert_distribution:
            return {"min": self.horiz_min, "max": self.horiz_max}
        else:
            return {"horiz_min": self.horiz_min, "horiz_max": self.horiz_max,
                    "vert_min": self.vert_min, "vert_max": self.vert_max}



class Bernoulli(Distribution):
    def __init__(self, prob=.5, do_rsample=False):
        super().__init__(do_rsample)
        self.prob = torch.autograd.Variable(torch.tensor([prob]))
        self.distribution = torch.distributions.Bernoulli(probs=self.prob)

    def __call__(self, size: TensorSize = 1):
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.distribution.sample(size).view(size)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        prob = self.prob.item()
        return f"{name}(prob={prob}, do_rsample={self.do_rsample})"

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Bernoulli(prob=self.prob.item(), do_rsample=do_rsample)

    def parameters(self):
        return {"prob": self.prob}


class Categorical(Distribution):
    def __init__(self, n_categories: int = 0, probs=(), do_rsample=False):
        super().__init__(do_rsample)
        if n_categories == 0:
            assert probs != ()
        else:
            assert probs == ()
            probs = tuple([1/n_categories for _ in range(n_categories)])

        self.probs = torch.autograd.Variable(torch.Tensor([probs]))
        self.distribution = torch.distributions.Categorical(probs=self.probs)

    def __call__(self, size: TensorSize = 1):
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.distribution.sample(size).view(size)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        probs = tuple(self.probs.tolist())
        return f"{name}(prob={probs}, do_rsample={self.do_rsample})"

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Categorical(probs=tuple(self.probs.tolist()), do_rsample=do_rsample)

    def parameters(self) -> dict:
        return {"probs": self.probs}


class Normal(Distribution):
    def __init__(self, mean=0.0, deviation=1.0, do_rsample=False):
        super.__init__(do_rsample=do_rsample)
        self.mean = torch.autograd.Variable(torch.Tensor([mean]))
        self.deviation = torch.autograd.Variable(torch.Tensor([deviation]))
        self.distribution = torch.distributions.Normal(loc=self.mean, scale=self.deviation)

    def __call__(self, size: TensorSize = 1):
        if not hasattr(size, "__getitem__"):
            size = [size]
        if self.do_rsample:
            raise self.distribution.rsample(size).view(size)
        else:
            return self.distribution.sample(size).view(size)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        params = f"mean={self.mean.item()}, deviation={self.deviation.item()}"
        return f"{name}({params}, do_rsample={self.do_rsample})"

    def copy(self, do_rsample):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Normal(probs=tuple(self.probs.tolist()), do_rsample=do_rsample)

    def parameters(self) -> dict:
        return {"probs": self.probs}
