import tormentor
import torch
import torchvision

ds = torchvision.datasets.CIFAR100(download="True", train=True,root="/tmp/downloads")
print(ds.train)