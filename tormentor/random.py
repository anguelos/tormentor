from typing import Tuple, Union

import torch

TupleRange = Tuple[float, float]
Tuple2DRange = Union[Tuple[float, float], Tuple[float, float, float, float]]  # min max, min_x, max_x, min_y max_y
TensorSize = Union[torch.Size, int, Tuple[int], Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]
ImageSize = Union[Tuple[int, int], Tuple[torch.LongTensor, torch.LongTensor]]


class Distribution(torch.nn.Module):
    def __init__(self, do_rsample):
        super().__init__()
        self.do_rsample = do_rsample

    def forward(self, size: TensorSize = 1, device="cpu"):
        raise NotImplementedError()

    def copy(self, do_rsample=None):
        raise NotImplementedError()

    def get_distribution_parameters(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError
        #this should be tested
        return type(self) is type(other) and self.get_distribution_parameters() == other.get_distribution_parameters()

    @property
    def device(self):
        raise NotImplementedError()

    def to(self, device):
        if device != self.device:
            super().to(device)

    def __hash__(self):
        return hash(tuple(sorted(self.get_distribution_parameters().items())))


class Uniform(Distribution):
    def __init__(self, value_range: TupleRange = (0.0, 1.0), do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.min = torch.nn.Parameter(torch.Tensor([value_range[0]]))
        self.max = torch.nn.Parameter(torch.Tensor([value_range[1]]))
        self.distribution = torch.distributions.Uniform(low=self.min, high=self.max)
        self.do_rsample = do_rsample

    def __repr__(self):
        range_str = f"({self.distribution.low.item()}, {self.distribution.high.item()})"
        param_str = f" do_rsample={self.do_rsample}"
        return f"{self.__class__.__qualname__}(value_range={range_str}, {param_str})"

    def __str__(self):
        range_str = f"({self.distribution.low.item():.3}, {self.distribution.high.item():.3})"
        return f"{self.__class__.__qualname__}(value_range={range_str})"


    def forward(self, size: TensorSize = 1, device="cpu") -> torch.Tensor:
        self.to(device)
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.distribution.sample(size).view(size).to(device)

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Uniform(value_range=(self.min.item(), self.max.item()), do_rsample=do_rsample)

    def get_distribution_parameters(self):
        return {"min": self.min, "max": self.max}

    @property
    def device(self):
        return self.min.device


class Constant(Distribution):
    def __init__(self, value: float, do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.value = torch.nn.Parameter(torch.Tensor([value]))

    def __repr__(self):
        value_str = tuple(self.value.detach().cpu().numpy())
        return f"{self.__class__.__qualname__}(value={value_str})"

    def forward(self, size: TensorSize = 1, device="cpu") -> torch.Tensor:
        self.to(device)
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = (size,)
            return self.value.repeat(size).to(device)

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Constant(value=self.value.item(), do_rsample=do_rsample)

    def get_distribution_parameters(self):
        return {"value": self.value}

    @property
    def device(self):
        return self.value.device

class Bernoulli(Distribution):
    def __init__(self, prob: float = .5, do_rsample: object = False) -> object:
        super().__init__(do_rsample)
        self.prob = torch.nn.Parameter(torch.tensor([prob]))
        self.distribution = torch.distributions.Bernoulli(probs=self.prob)

    def forward(self, size: TensorSize = 1, device="cpu"):
        self.to(device)
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.distribution.sample(size).view(size).to(device)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        prob = self.prob.cpu().tolist()
        return f"{name}(prob={repr(prob)}, do_rsample={self.do_rsample})"

    def __str__(self) -> str:
        name = self.__class__.__qualname__
        prob = float(self.prob.cpu()[0])
        return f"{name}(prob={repr(prob):.3})"

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Bernoulli(prob=self.prob.item(), do_rsample=do_rsample)

    def get_distribution_parameters(self):
        return {"prob": self.prob}

    @property
    def device(self):
        return self.prob.device

class Categorical(Distribution):
    def __init__(self, n_categories: int = 0, probs=(), do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        if n_categories == 0:
            assert probs != ()
        else:
            assert probs == ()
            probs = tuple([1/n_categories for _ in range(n_categories)])

        self.probs = torch.autograd.Variable(torch.Tensor([probs]))
        self.distribution = torch.distributions.Categorical(probs=self.probs)

    def forward(self, size: TensorSize = 1, device="cpu"):
        self.to(device)
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, "__getitem__"):
                size = [size]
            return self.distribution.sample(size).view(size).to(device)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        probs = tuple(self.probs.tolist())
        return f"{name}(prob={probs}, do_rsample={self.do_rsample})"

    def __str__(self) -> str:
        name = self.__class__.__qualname__
        probs = "(" + ", ".join([f"{p:.3}" for p in self.probs.tolist()]) + ")"
        return f"{name}(prob={probs})"

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Categorical(probs=tuple(self.probs.tolist()), do_rsample=do_rsample)

    def get_distribution_parameters(self) -> dict:
        return {"probs": self.probs}

    @property
    def device(self):
        return self.probs.device


class Normal(Distribution):
    def __init__(self, mean=0.0, deviation=1.0, do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.mean = torch.nn.Parameter(torch.Tensor([mean]))
        self.deviation = torch.nn.Parameter(torch.Tensor([deviation]))
        self.distribution = torch.distributions.Normal(loc=self.mean, scale=self.deviation)

    def forward(self, size: TensorSize = 1, device="cpu"):
        self.to(device)
        if not hasattr(size, "__getitem__"):
            size = [size]
        if self.do_rsample:
            raise self.distribution.rsample(size).view(size).to(device)
        else:
            return self.distribution.sample(size).view(size).to(device)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        params = f"mean={self.mean.item()}, deviation={self.deviation.item()}"
        return f"{name}({params}, do_rsample={self.do_rsample})"

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Normal(mean=self.mean, deviation=self.deviation, do_rsample=do_rsample)

    def get_distribution_parameters(self) -> dict:
        return {"mean": self.mean, "deviation": self.deviation}

    @property
    def device(self):
        return self.mean.device