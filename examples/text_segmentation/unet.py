import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels,preserve=False):
        super().__init__()
        self.conv2out=nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
        )
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
        )
        self.preserve=preserve

    def forward(self, x):
        if self.preserve:
            x = self.conv2out(x)
            #print(self.double_conv1(x).size(),x.size())
            new_x = x.clone()
            new_x[:,:,1:-1,1:-1] = x[:,:,1:-1,1:-1] + self.double_conv1(x)
            x=new_x
            new_x = x.clone()
            new_x[:,:,1:-1,1:-1] = x[:,:,1:-1,1:-1] + self.double_conv2(x)
            return x
        else:
            x = self.conv2out(x)
            #print(self.double_conv1(x).size(),x.size())
            x = x[:,:,1:-1,1:-1] + self.double_conv1(x)
            x = x[:,:,1:-1,1:-1] + self.double_conv2(x)
            return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels,preserve=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64,preserve=True)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_device(self):
        return self.outc.conv.weight.device

    def __repr__(self):
        return "{}(n_channels={}, n_classes={}, bilinear={})".format(repr(self.__class__),repr(self.n_channels),repr(self.n_classes),repr(self.bilinear))

    def save(self,path):
        save_dict = self.state_dict()
        save_dict["model"] = repr(self)
        torch.save(self.save_dict(), path)

    @staticmethod
    def load(path):
        loaded_dict= torch.load(path)
        model=eval(loaded_dict["model"])
        del loaded_dict["model"]
        model.load_state_dict(loaded_dict, strict=False)
        return model