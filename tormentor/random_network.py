from .random import Distribution
import torch


def _seed_2_float_bitvector(seeds=torch.LongTensor, bit_count=64) -> torch.FloatTensor:
    assert bit_count <= 64
    res = torch.empty([seeds.size(0), bit_count], device=seeds.device)
    for n in range(1, bit_count):
        res[:, n] = (seeds // n) % 2
    return res


class DistributionNetwork(torch.nn.Module):
    def __init__(self, sizes=[64, 128, 1], dropout=.8, categorical=True):
        super().__init__()
        self.input_size = sizes[0]
        layers = [torch.nn.Linear(sizes[0])]
        for n in range(1, len(sizes)-1):
            layers.append(torch.nn.Linear(sizes[n-1], sizes[n]))
            if dropout > 0.:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        if categorical:
            layers.append(torch.nn.Softmax())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x_seeds=None, x_rnd=None, size=None):
        if size is not None and x_seeds is None and x_rnd is None:
            x = torch.rand(size=size)
        elif size is None and x_seeds is not None and x_rnd is None:
            x = _seed_2_float_bitvector(x_seeds, self.input_size)
        elif size is None and x_seeds is None and x_rnd is not None:
            x = x_rnd
        return self.layers(x)



class DistributionNetwork(Distribution):
    def __init__(self):
        super().__init__()

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
