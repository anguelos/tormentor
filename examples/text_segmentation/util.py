import numpy as np
import torch
import fastai
import fastai.vision
from unet import UNet
from unet2 import  R2AttU_Net, AttU_Net, R2U_Net, U_Net
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import types
import PIL

class SingleImageDataset(object):
    def __init__(self, image_filename_list, cache_images=True):
        self.image_filenames = image_filename_list
        self.cache_images = cache_images
        if self.cache_images:
            self.cache=[]
            for filename in self.image_filenames:
                self.cache.append(PIL.Image.open(filename))

    def __getitem__(self, item):
        if self.cache_images:
            return self.cache[item]
        elif self.last_img_idx == item:
            return self.last_img
        else:
            img = PIL.Image.open(self.image_filenames[item])
            self.last_img = img
            return self.last_img


class TiledDataset(object):
    def _create_idx(self):
        image_in_sample = self.is_image_in_sample.find(True)
        patch_idx = 0
        self.patch_idx = []
        self.sample_idx = []
        self.left = []
        self.top = []
        self.sizes=[]
        self.inverse_starts = []
        self.inverse_ends=[]
        total_counter=0
        for img_idx,sample in enumerate(self.dataset):
            self.inverse_starts.append(total_counter)
            img = sample[image_in_sample]
            sz = img.size
            self.sizes.append(sz)
            self.patch_idx = 0
            for horiz in range(0, sz[0], self.tile_size[0]):
                for vert in range(0, sz[1], self.tile_size[1]):
                    self.left.append(horiz)
                    self.top.append(vert)
                    self.patch_idx.append(patch_idx)
                    self.sample_idx.append(img_idx)
                    patch_idx += 1
                    total_counter += 1
            self.inverse_ends.append(total_counter)

    def __init__(self, dataset, is_image_in_sample=None, tile_size=256, ltrb_pad=(64,64,64,64), input_transform=lambda x:x, output_transform=lambda x:x):
        if is_image_in_sample is None:
            is_image_in_sample = [isinstance(datum, PIL.Image) for datum in dataset[0]]
        self.dataset = dataset
        self.is_image_in_sample = is_image_in_sample
        self.tile_size = tile_size
        self.ltrb_pad = ltrb_pad
        self._create_idx()

    def __getitem__(self, item):
        sample = self.ds[self.sample_idx[item]]
        res_sample = []
        for n, datum in enumerate(sample):
            if self.is_image_in_sample[n]:
                patch = datum.crop((self.left[item]-self.ltrb_pad[0], self.top[item]-self.ltrb_pad[1], self.left[item] + self.tile_size[0]+self.ltrb_pad[1], self.top[item] + self.tile_size[1]+self.ltrb_pad[1]))
                res_sample.append(patch)
            else:
                res_sample.append(datum)
        return res_sample

    def __len__(self):
        return len(self.sample_idx)

    def sample_as_list(self, n_sample):
        res=[]
        for n in range(self.inverse_starts[n_sample], self.inverse_ends[n_sample]):
            res.append(self[n])
        return res

    def stich_image_tensors(self, image_list, sz):
        image_stack = image_list.copy()
        out_image = []
        for horiz in range(0, sz[0], self.tile_size[0]):
            column = []
            for vert in range(0, sz[1], self.tile_size[1]):
                patch = image_stack.pop(0)
                left,top,right,bottom = self.ltrb_pad[0], self.ltrb_pad[1],patch.size(2)-self.ltrb_pad[2], patch.size(3)-self.ltrb_pad[3]
                patch = patch[:,top:bottom,left:right]
                column.append(patch)
            column=torch.cat(column,dim=2)
            out_image.append(column)
        out_image = torch.cat(out_image, dim=3)
        return out_image[:, :sz[1], :sz[0]]


    def apply_network(self, network, sample_pos=0, datum_idx=0):
        images = [sample[datum_idx] for sample in self.sample_as_list(sample_pos)]


def modified_forward(self, input_x):
    if self.padding[0] > 0 or self.padding[1] > 0:
        batch_size, n_channels, height, width = input_x.size()
        #x2d = input_x.view([batch_size * n_channels, height * width])
        x2d = input_x.reshape([batch_size * n_channels, height * width])
        mean = x2d.mean(dim=1).view([batch_size, n_channels, 1, 1])
        std = x2d.std(dim=1).view([batch_size, n_channels, 1, 1]) / (2*height+2*width)
        n_pad_left = self.padding[0]
        n_pad_top = self.padding[1]
        n_pad_right = self.padding[0]
        n_pad_bottom = self.padding[1]
        pad_left = torch.normal(mean.repeat((1,1, height, n_pad_left)), std.repeat((1, 1, height, n_pad_left)))
        pad_right = torch.normal(mean.repeat(1,1, height, n_pad_right), std.repeat(1, 1, height, n_pad_right))
        pad_top = torch.normal(mean.repeat(1,1, n_pad_top, n_pad_left + width + n_pad_right), std.repeat(1, 1, n_pad_top, n_pad_left + width + n_pad_right))
        pad_bottom = torch.normal(mean.repeat(1,1, n_pad_bottom, n_pad_left + width + n_pad_right), std.repeat(1, 1, n_pad_bottom, n_pad_left + width + n_pad_right))
        input_x = torch.cat([pad_top, torch.cat([pad_left, input_x, pad_right], dim=3), pad_bottom], dim=2)
    output_x = F.conv2d(input_x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
    return output_x


def patch_2d_conv(conv2d_layer):
    conv2d_layer.forward = types.MethodType(modified_forward, conv2d_layer)

def patch_all_2d_conv(module):
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.forward = types.MethodType(modified_forward, layer)




def patch_forward(net, img, patch_width, patch_height):
    padded_img = torch.zeros([img.size(0), img.size(1), (1+img.size(2)//patch_height)*patch_height,(1+img.size(3)//patch_width)*patch_width],device=img.device)
    print("Padded:", padded_img.size())
    padded_img[:, :, :img.size(2), :img.size(3)]=img
    res = torch.zeros([img.size(0), 2, padded_img.size(2), padded_img.size(3)], device=img.device)
    for left in range(0, padded_img.size(3), patch_width):
        for top in range(0, padded_img.size(2), patch_height):
            patch = padded_img[:, :, top:top+patch_height, left:left+patch_width]
            print("patch:",patch.size())
            out_patch = net(patch)
            res[:, :, top:top+patch_height, left:left+patch_width] = out_patch
    return res[:, :, :res.size(2), :res.size(3)]


def create_net(arch,n_channels,n_classes, bn_momentum,rnd_pad, pretrained=True):
    if arch == "dunet34":
        print("Creating Resnet")
        body = fastai.vision.learner.create_body(fastai.vision.models.resnet34, pretrained=pretrained, cut=-2)
        print("Creating Unet")
        net = fastai.vision.models.unet.DynamicUnet(body, n_classes)
        print("Done")
        if bn_momentum <0:
            bn_momentum=None
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = bn_momentum
    elif arch == "dunet50":
        print("Creating Resnet")
        body = fastai.vision.learner.create_body(fastai.vision.models.resnet50, pretrained=pretrained, cut=-2)
        print("Creating Unet")
        net = fastai.vision.models.unet.DynamicUnet(body, n_classes)
        print("Done")
        if bn_momentum <0:
            bn_momentum=None
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = bn_momentum
    elif arch=="dunet18":
        print("Creating Resnet")
        body = fastai.vision.learner.create_body(fastai.vision.models.resnet18, pretrained=pretrained, cut=-2)
        print("Creating Unet")
        net = fastai.vision.models.unet.DynamicUnet(body, n_classes)
        print("Done")
        if bn_momentum <0:
            bn_momentum=None
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = bn_momentum
    elif arch== "unet":
        net = UNet(n_channels=n_channels, n_classes=n_classes)
    elif arch == "R2AttUNet":
        net = R2AttU_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "AttUNet":
        net = AttU_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "R2UNet":
        net = R2U_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "UNet":
        #img_ch=3,output_ch=1
        net = U_Net(img_ch=n_channels, output_ch=n_classes)
    else:
        raise ValueError("arch must be either dunet34, dunet50, dunet18, or unet")
    if rnd_pad:
        patch_all_2d_conv(net)
    return net


def save(param_hist, per_epoch_train_errors,per_epoch_validation_errors,epoch,net):
    p=param_hist[sorted(param_hist.keys())[-1]]
    save_dict=net.state_dict()
    save_dict["param_hist"]=param_hist
    save_dict["per_epoch_train_errors"]=per_epoch_train_errors
    save_dict["per_epoch_validation_errors"] = per_epoch_validation_errors
    save_dict["epoch"]=epoch
    torch.save(save_dict, p["resume_fname"])
    if p["archive_nets"]:
        folder="/".join(p["resume_fname"].split("/")[:-1])
        if folder == "":
            folder = "."
        torch.save(save_dict, f"{folder}/{p['arch']}_{epoch:05}.pt")

def resume(net,resume_fname, device):
    try:
        save_dict=torch.load(resume_fname,map_location=device)
        if "param_hist" in save_dict.keys():
            param_hist=save_dict["param_hist"]
            del save_dict["param_hist"]
        else:
            param_hist={}
        per_epoch_train_errors=save_dict["per_epoch_train_errors"]
        del save_dict["per_epoch_train_errors"]
        per_epoch_validation_errors=save_dict["per_epoch_validation_errors"]
        del save_dict["per_epoch_validation_errors"]
        start_epoch=save_dict["epoch"]
        del save_dict["epoch"]
        net.load_state_dict(save_dict)
        print("Resumed from ",resume_fname)
        return param_hist, per_epoch_train_errors, per_epoch_validation_errors,start_epoch,net
    except FileNotFoundError as e:
        print("Failed to resume from ", resume_fname)
        return {}, {}, {}, 0, net


def render_confusion(prediction,gt,valid_mask=None,tp_col=[0, 0, 0],tn_col=[255,255,255],fp_col=[255,0,0],fn_col=[0,0,255], undetermined_col=[128,128,128]):
    prediction=(prediction.cpu().numpy()>.5)
    gt = (gt.cpu().numpy()>.5)
    res=np.zeros(prediction.shape+(3,))
    if valid_mask is not None:
        valid_mask = valid_mask.cpu().numpy()
        tp = gt & prediction & valid_mask
        tn = (~gt) & (~prediction) & valid_mask
        fp = (~gt) & prediction & valid_mask
        fn = (gt) & (~prediction) & valid_mask
    else:
        tp = gt & prediction
        tn = (~gt) & (~prediction)
        fp = (~gt) & prediction
        fn = (gt) & (~prediction)
        valid_mask = np.ones_like(gt)
    res[~valid_mask, :] = undetermined_col
    res[tp, :] = tp_col
    res[tn, :] = tn_col
    res[fp, :] = fp_col
    res[fn, :] = fn_col
    precision = (1+tp.sum())/float(1+tp.sum()+fp.sum())
    recall = (1+tp.sum()) / float(1+tp.sum() + fn.sum())
    Fscore=(2*precision*recall)/(precision+recall)
    return res, precision, recall, Fscore