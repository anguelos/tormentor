import torch
import torch.nn.functional as F
import types


def modified_forward(self, x):
    x = F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

    batch_size, n_channels, height, width = x.size()
    x2d = x.view([batch_size * n_channels, height * width])
    mean = x2d.mean(dim=1).view([batch_size, n_channels, 1, 1])
    std = x2d.std(dim=1).view([batch_size, n_channels, 1, 1])
    n_pad_left = self.padding[0]
    n_pad_top = self.padding[1]
    n_pad_right = self.padding[0]
    n_pad_bottom = self.padding[1]
    pad_left = torch.normal(mean.repeat((1,1, height, n_pad_left)), std.repeat((1, 1, height, n_pad_left)))
    pad_right = torch.normal(mean.repeat(1,1, height, n_pad_right), std.repeat(1, 1, height, n_pad_right))
    pad_top = torch.normal(mean.repeat(1,1, n_pad_top, n_pad_left + width + n_pad_right), std.repeat(1, 1, n_pad_top, n_pad_left + width + n_pad_right))
    pad_bottom = torch.normal(mean.repeat(1,1, n_pad_bottom, n_pad_left + width + n_pad_right), std.repeat(1, 1, n_pad_bottom, n_pad_left + width + n_pad_right))
    x = torch.cat([pad_top, torch.cat([pad_left, x, pad_right], dim=3), pad_bottom], dim=2)
    return x


def patch_2d_conv(conv2d_layer):
    conv2d_layer.forward = types.MethodType(modified_forward, conv2d_layer)

def patch_all_2d_conv(module):
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.forward = types.MethodType(modified_forward, layer)


