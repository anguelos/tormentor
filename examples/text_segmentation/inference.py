#!/usr/bin/env python3

#https://github.com/pytorch/pytorch/issues/29893
import os
os.environ["LRU_CACHE_CAPACITY"] = "1"

import pathlib
import time
import PIL
import fargv
import fastai.vision
import torch
import torchvision
from unet import UNet

p = {
    "n_channels":3,
    "n_classes":2,
    "bn_momentum":.1,
    "arch":["dunet","unet"],
    "device": "cpu",
    "model_fname": "binnet_normal.pt",
    "inputs": set([]),
    "n_classes": 2,
    "flip": [True, "Transpose image axes"],
    "recompute": False,
    "text_is_white": True,
    "out_filename_prefix": "res_",
    "out_filename_suffix": "",
    "out_filetype": "png",
}

p, _ = fargv.fargv(p, return_named_tuple=True)


def resume(net):
    try:
        save_dict = torch.load(p.model_fname, map_location=p.device)
        param_hist = save_dict["param_hist"]
        del save_dict["param_hist"]
        per_epoch_train_errors = save_dict["per_epoch_train_errors"]
        del save_dict["per_epoch_train_errors"]
        per_epoch_validation_errors = save_dict["per_epoch_validation_errors"]
        del save_dict["per_epoch_validation_errors"]
        start_epoch = save_dict["epoch"]
        del save_dict["epoch"]
        net.load_state_dict(save_dict)
        print("Resumed from ", p.model_fname)
        return param_hist,per_epoch_train_errors, per_epoch_validation_errors, start_epoch, net
    except FileNotFoundError as e:
        print("Failed to resume from ", p.resume_fname)
        return {}, {}, 0, net


device = torch.device(p.device)
if p.arch == "dunet":
    print("Creating Resnet")
    body = fastai.vision.learner.create_body(fastai.vision.models.resnet34, pretrained=True, cut=-2)
    print("Creating Unet")
    net = fastai.vision.models.unet.DynamicUnet(body, 2)
    print("Done")
    if p.bn_momentum <0:
        p.bn_momentum=None
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = p.bn_momentum
elif p.arch== "unet":
    net = UNet(n_channels=p.n_channels, n_classes=p.n_classes)
else:
    raise ValueError("arch must be either dunet or unet")

print("Resuming Model ... ", end="")
param_hist,per_epoch_train_errors, per_epoch_validation_errors, start_epoch, net = resume(net)
print("done!")
print("Computing images:")

in_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
out_transform = torchvision.transforms.ToPILImage()


total_start=time.time()
with torch.no_grad():
    for n,filename in enumerate(p.inputs):
        print(filename, end="")
        t = time.time()
        out_fname = filename.split("/")
        striped_name = out_fname[-1].split(".")[0]
        out_fname[-1] = p.out_filename_prefix + striped_name + p.out_filename_suffix + "." + p.out_filetype
        out_fname = "/".join(out_fname)
        if p.recompute or not pathlib.Path(out_fname).exists():
            img = PIL.Image.open(filename)
            img = in_transform(img)
            if p.flip:
                img = img.transpose(1, 2)
            img = img.unsqueeze(dim=0).to(p.device)
            # padding image to even pixels
            pwidth = img.size(3) + img.size(3) % 2
            pheight = img.size(2) + img.size(2) % 2
            padded_img = torch.nn.functional.pad(input=img, pad=(0, 1, 0, 1))
            padded_img = padded_img[:, :, :pheight, :pwidth]
            prediction = net(padded_img)
            res = torch.nn.functional.softmax(prediction, dim=1)
            if not p.text_is_white:
                res = res[:, [1, 0], :, :]
            res = res[0, :, :img.size(2), :img.size(3)].cpu().detach()
            if p.flip:
                res = res.transpose(1, 2)
            res = (res[0, :, :] < res[1, :, :]).numpy()
            res_img = PIL.Image.fromarray(res.astype("uint8") * 255).convert('RGB')
            print(f" {(res.shape[0] * res.shape[0] / 1000000):.3} MP in {time.time() - t:.5} sec.")
            res_img.save(out_fname)
        else:
            print(" found ", out_fname, " keeping it")