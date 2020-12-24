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
import PIL
from unet import UNet
from util import render_confusion,resume,create_net, patch_forward
import pickle

p = {
    "n_channels":3,
    "n_classes":2,
    "bn_momentum":.1,
    "arch":[("dunet34","dunet18","dunet50","unet", "R2AttUNet", "AttUNet", "R2UNet", "UNet"),"Model Archtecture"],
    "device": "cpu",
    "model_fname": "{arch}.pt",
    "inputs": set([]),
    "flip": [True, "Transpose image axes"],
    "recompute": False,
    "text_is_white": True,
    "out_filename_prefix": "res_",
    "out_filename_suffix": "",
    "out_filetype": "png",

    "gt_filename_prefix": "gt_",
    "gt_filename_suffix": "",
    "gt_filetype": "png",

    "eval_filename_prefix": "eval_",
    "eval_filename_suffix": "",
    "eval_filetype": "png",

    "eval_store_file": "/tmp/eval.pickle",
    "rnd_pad": True,
    "crop": 0,
    "eval": False,
    "patch_width":-1,
    "patch_height":-1
}

p, _ = fargv.fargv(p, return_named_tuple=True)



#device = torch.device(p.device)

net = create_net(p.arch, p.n_channels, p.n_classes, p.bn_momentum,p.rnd_pad)

print("Resuming Model ... ", end="")
param_hist, per_epoch_train_errors, per_epoch_validation_errors, start_epoch, net = resume(net, p.model_fname, p.device)
net=net.to(p.device)
print("done!")
print("Computing images:")

in_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def img2gt(x):
    return (x<.99).float()
gt_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor(), img2gt])
out_transform = torchvision.transforms.ToPILImage()

all_res={}
total_start=time.time()
with torch.no_grad():
    for n, filename in enumerate(p.inputs):
        print(filename, end="")
        t = time.time()
        out_fname = filename.split("/")
        striped_name = out_fname[-1].split(".")[0]
        out_fname[-1] = p.out_filename_prefix + striped_name + p.out_filename_suffix + "." + p.out_filetype
        out_fname = "/".join(out_fname)
        if p.eval:
            gt_fname = filename.split("/")
            striped_name = gt_fname[-1].split(".")[0]
            gt_fname[-1] = p.gt_filename_prefix + striped_name + p.gt_filename_suffix + "." + p.gt_filetype
            gt_fname = "/".join(gt_fname)

            eval_fname = filename.split("/")
            striped_name = eval_fname[-1].split(".")[0]
            eval_fname[-1] = p.eval_filename_prefix + striped_name + p.eval_filename_suffix + "." + p.eval_filetype
            eval_fname = "/".join(eval_fname)

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
            if p.patch_width>0 and p.patch_height:
                prediction = patch_forward(net, padded_img, p.patch_width, p.patch_height)[:,:,:img.size(2), :img.size(3)]
            else:
                prediction = net(padded_img)[:,:,:img.size(2), :img.size(3)]
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            res = torch.cat([prediction,1-prediction],dim=1).detach()

            #res = torch.nn.functional.softmax(prediction, dim=1)

            #prediction = net(padded_img)[:,:,:img.size(2), :img.size(3)]

            if not p.text_is_white:
                res = res[:, [1, 0], :, :]
            res = res[0, :, :img.size(2), :img.size(3)].cpu().detach()
            if p.flip:
                res = res.transpose(1, 2)
            res = (res[0, :, :] < .5)
            if p.eval:
                gt = gt_transform(PIL.Image.open(gt_fname))
                gt = gt[0,:,:]>0

                confusion, precision, recall, fscore = render_confusion(res, gt)
                confusion = confusion.astype("uint8")
                #print("confusion",confusion.shape,confusion.dtype,eval_fname)
                all_res[filename]=(precision,recall,fscore)
                print(f"{n:3}, {time.time()-t:6.3} {time.time()-total_start:8.3} PR:{precision}, R:{recall}, F:{fscore}")
                PIL.Image.fromarray(confusion).save(eval_fname)

            res = res.numpy()

            res_img = PIL.Image.fromarray(res.astype("uint8") * 255).convert('RGB')
            #print(f" {(res.shape[0] * res.shape[0] / 1000000):.3} MP in {time.time() - t:.5} sec.")
            res_img.save(out_fname)
        else:
            print(" found ", out_fname, " keeping it")
        pickle.dump(all_res,open(p.eval_store_file, "wb"))