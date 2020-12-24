#!/usr/bin/env python3

import PIL
import fastai.vision
import tormentor
import glob
from tormentor import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import torch
import fargv
import torchvision

from rr_ds import RR2013Ch2, augment_RRDS_batch
from dibco import Dibco, dibco_transform_gt, dibco_transform_gt, dibco_transform_color_input

from util import render_confusion, save, resume,create_net
import cv2
import time
import tqdm
import sys


torch.multiprocessing.set_start_method('spawn', force=True)

torch.autograd.set_detect_anomaly(True)

p={
    "n_channels":3,
    "n_classes":2,
    "save_images":False,
    "save_input_images":False,
    "self_pattern": "*png",
    "arch":[("dunet34", "dunet18", "dunet50", "unet", "R2AttUNet", "AttUNet", "R2UNet", "UNet"), "Model Archtecture"],
    "rrds_root": "/home/anguelos/data/rr/focused_segmentation/zips",
    "dataset": [("rrds", "self_eval_2013", "dibco2010", "dibco2011", "dibco2012", "dibco2013", "dibco2014", "dibco2016", "dibco2017", "dibco2018", "dibco2019"), "Either Robust Reading Segmentation (rrds), or Document Image Binarization"],
    #"augmentation": "tormentor.RandomPlasmaBrightness & tormentor.RandomPerspective & tormentor.RandomHue & tormentor.RandomSaturation & tormentor.RandomInvert",
    #"val_augmentation": "CropPadTo.new_size({patch_width}, {patch_height})",
    "val_augmentation": "",
    "train_augmentation": "(RandomPlasmaLinearColor & RandomWrap.custom(roughness=Uniform(value_range=(0.1, 0.7)), intensity=Uniform(value_range=(0.18, 0.62))) & RandomPlasmaShadow.custom(roughness=Uniform(value_range=(0.334, 0.72)), shade_intencity=Uniform(value_range=(-0.32, 0.0)), shade_quantity=Uniform(value_range=(0.0, 0.44))) & RandomPerspective.custom(x_offset=Uniform(value_range=(0.75, 1.0)), y_offset=Uniform(value_range=(0.75, 1.0))) & RandomPlasmaBrightness.custom(roughness=Uniform(value_range=(0.1, 0.4)),intencity=Uniform(value_range=(0.322, 0.9))) )",
    #"train_augmentation": "(RandomSaturation.custom(saturation=Constant(0.0)) & PadCropTo.new_size({patch_width}, {patch_height}))",
    #"val_augmentation": "(RandomWrap.custom(roughness=Uniform(value_range=(0.1, 0.7)), intensity=Uniform(value_range=(0.18, 0.62))) & RandomPlasmaShadow.custom(roughness=Uniform(value_range=(0.334, 0.472)), shade_intencity=Uniform(value_range=(-0.32, 0.0)), shade_quantity=Uniform(value_range=(0.0, 0.44))) & RandomPerspective.custom(x_offset=Uniform(value_range=(0.75, 1.0)), y_offset=Uniform(value_range=(0.75, 1.0))) & RandomPlasmaBrightness.custom(roughness=Uniform(value_range=(0.322, 0.4))) & RandomSaturation.custom(saturation=Constant(0.0)) & PadCropTo.new_size(512, 512))",
    "io_threads": 1,
    "log_freq": 10,
    "lr": .001,
    "epochs": 10,
    "tormentor_device":"cpu",
    "device":"cuda",
    "val_device":"{device}",
    "validate_freq": 5,
    "trainoutputs_freq": 5,
    "archive_nets":False,
    "batch_size": 1,
    "save_freq": 10,
    "mask_gt": 1,
    "resume_fname":"{arch}.pt",
    "patch_width": 512,
    "patch_height": 512,
    "val_patch_width": "{patch_width}",
    "val_patch_height": "{patch_width}",
    "rnd_pad":False,
    "crop_loss":0,
    "pretrained":True,
    "bn_momentum":(.1, "[0.-1.] negative for None this changes the bathnormisation momentum parameter.")
}
param_dict, _ = fargv.fargv(p.copy(), argv=sys.argv.copy(), return_named_tuple=False)
p, _ = fargv.fargv(p, return_named_tuple=True)
device = torch.device(p.device)

def run_epoch(p,device,loader,net,criterion,optimizer=None,save_images=True,is_deeplabv3=True, save_input_images=False):
    is_validation = optimizer is None
    net.to(device)
    if is_validation:
        isval_str = "Validation"
        do_grad=lambda: torch.no_grad()
        net.eval()
    else:
        do_grad = lambda: open("/tmp/fake","w")
        isval_str = "Train"
        net.train()
    fscores = []
    precisions = []
    recalls = []
    losses = []
    model_outputs={}
    t=time.time()
    with do_grad():
        for n, (input_img,gt,mask) in tqdm.tqdm(enumerate(loader)):
            input_img, gt, mask = input_img.to(device), gt.to(device), mask.to(device)
            coeffs = mask.unsqueeze(dim=0)
            #f, ax = plt.subplots(3,1)
            #print("input_img",input_img.size())
            #print("gt",gt.size())
            #print("mask",mask.size())
            #ax[0].imshow(input_img[0, :, :, :].detach().cpu().transpose(0,2).transpose(0,1),vmin=0.,vmax=1.)
            #ax[1].imshow(gt[0, 1, :, :].detach().cpu(),vmin=0.,vmax=1., cmap="gray")
            #ax[2].imshow(mask[0, 0, :, :].detach().cpu(),vmin=0.,vmax=1., cmap="gray")
            #plt.show()
            #print("quiting")
            #sys.exit()

            #coeffs = gt[:, 1, :, :].mean(dim=1).mean(dim=1)
            #coeffs = coeffs.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            #coeffs = coeffs * mask
            # ignoring fg ratio from loss
            #coeffs = torch.ones_like(coeffs) * mask

            if p.crop_loss>0:
                coeffs[:,:,:p.crop_loss,:0]=0
                coeffs[:,:,-p.crop_loss:,:0]=0
                coeffs[:,:,:,p.crop_loss]=0
                coeffs[:,:,:,-p.crop_loss:]=0
            #print("input_img", input_img.size())
            prediction = net(input_img)
            #print("PR GT", prediction.size(), gt.size())
            loss = criterion(prediction, gt)
            loss = loss * coeffs
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            prediction = torch.cat([prediction, 1-prediction], dim=1).detach()
            loss=loss.sum()
            if not is_validation:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            #bin_prediction = prediction[0,0,:,:]<get_otsu_threshold(prediction[0,0,:,:])
            #confusion,precision,recall,fscore = render_confusion(bin_prediction, gt[0, 0, :, :]<.5)
            confusion, precision, recall, fscore = render_confusion(prediction[0,0,:,:]<prediction[0,1,:,:], gt[0, 0, :, :] < .5, mask[0,0,:,:] > .5)
            #confusion,precision,recall,fscore = render_optimal_confusion(prediction[0,0,:,:], gt[0, 0, :, :]<.5)

            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)
            losses.append(loss.item()/gt.view(-1).size()[0])
            if save_images:
                cv2.imwrite("/tmp/{}_img_{}.png".format(isval_str,n), confusion)
                model_outputs[n]=prediction.detach().cpu()
            if save_input_images:
                torchvision.transforms.ToPILImage()(input_img[0,:,:,:].detach().cpu()).save(f"/tmp/input_{isval_str}_img_{n}.png")

    lines = []
    lines.append("Epoch {} {} Total:\t{:05f}%".format(epoch, isval_str, 100*sum(fscores)/(.0000001+len(fscores))))
    lines.append('')
    print("N:\t{} % computed in {:05f} sec.".format(isval_str, time.time()-t))
    print("\n".join(lines))
    #if save_images:
    #    torch.save(model_outputs,"/tmp/{}_samples.pt".format(isval_str))
    return sum(fscores) / len(fscores),sum(precisions) / len(precisions), sum(recalls) / len(recalls),sum(losses) / len(losses)


if p.dataset == "rrds":
    trainset = RR2013Ch2(train=True, return_mask=True,cache_ds=True,root=p.rrds_root,default_width=p.patch_width,default_height=p.patch_height)
    validationset = RR2013Ch2(train=False, return_mask=True, cache_ds=True, root=p.rrds_root, default_width=p.patch_width, default_height=p.patch_height)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=p.batch_size, num_workers= p.io_threads, drop_last=True)
    valloader = torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=1)
    if p.train_augmentation != "":
        train_augmentation_cls = eval(p.train_augmentation)
        trainloader = tormentor.AugmentedDataLoader(trainloader, train_augmentation_cls, torch.device(p.tormentor_device), augment_RRDS_batch)
        #test_augmentation_cls = eval(p.test_augmentation)
    if p.val_augmentation != "":
        val_augmentation_cls = eval(p.train_augmentation)
        valloader = tormentor.AugmentedDataLoader(valloader, val_augmentation_cls, torch.device(p.tormentor_device), augment_RRDS_batch)

elif p.dataset == "self_eval_2013":
    img_fnames = glob.glob(p.self_pattern)
    class SelfAugmentationDataset():
        def __init__(self,img_fnames):
            self.img_fnames = img_fnames
            self.input_transform = dibco_transform_color_input
            self.gt_transform = dibco_transform_gt

        def __getitem__(self, item):
            img = PIL.Image.open(self.img_fnames[item])
            input = self.input_transform(img)
            gt = self.gt_transform(img)
            mask = torch.ones_like(gt[:1,:,:])
            return input, gt, mask

        def __len__(self):
            return len(self.img_fnames)

    trainset = SelfAugmentationDataset(img_fnames)
    train_resizer= tormentor.PadCropTo.new_size(p.patch_width, p.patch_height)
    trainset = tormentor.AugmentedDs(trainset, augmentation_factory=train_resizer, computation_device="cpu", add_mask=False)

    validationset = Dibco.Dibco2013()
    validationset = tormentor.AugmentedDs(validationset, augmentation_factory=train_resizer, computation_device="cpu", add_mask=True)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=p.batch_size, num_workers= p.io_threads, drop_last=True)
    valloader = torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=1)

    if p.train_augmentation != "":
        augmentation_factory = tormentor.AugmentationFactory(eval(p.train_augmentation))
        trainloader = tormentor.AugmentedDataLoader(trainloader, augmentation_factory, torch.device(p.tormentor_device), augment_RRDS_batch)


    if p.val_augmentation != "":
        augmentation_factory = tormentor.AugmentationFactory(eval(p.val_augmentation))
        valloader = tormentor.AugmentedDataLoader(valloader, augmentation_factory, torch.device(p.tormentor_device), augment_RRDS_batch)

elif p.dataset in ["dibco2010", "dibco2011", "dibco2012", "dibco2013", "dibco2014", "dibco2016", "dibco2017", "dibco2018", "dibco2019"]:
    if p.dataset == "dibco2010":
        trainset = Dibco.Dibco2009()
        validationset = Dibco.Dibco2010()
    elif p.dataset == "dibco2011":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010()
        validationset = Dibco.Dibco2011()
    elif p.dataset == "dibco2012":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011()
        validationset = Dibco.Dibco2012()
    elif p.dataset == "dibco2013":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011() + Dibco.Dibco2012()
        validationset = Dibco.Dibco2013()
    elif p.dataset == "dibco2014":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011() + Dibco.Dibco2012() + Dibco.Dibco2013()
        validationset = Dibco.Dibco2014()
    elif p.dataset == "dibco2016":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011() + Dibco.Dibco2012() + Dibco.Dibco2013() + Dibco.Dibco2014()
        validationset = Dibco.Dibco2016()
    elif p.dataset == "dibco2017":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011() + Dibco.Dibco2012() + Dibco.Dibco2013() + Dibco.Dibco2014() + Dibco.Dibco2016()
        validationset = Dibco.Dibco2017()
    elif p.dataset == "dibco2018":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011() + Dibco.Dibco2012() + Dibco.Dibco2013() + Dibco.Dibco2014() + Dibco.Dibco2016() + Dibco.Dibco2017()
        validationset = Dibco.Dibco2018()
    elif p.dataset == "dibco2019":
        trainset = Dibco.Dibco2009() + Dibco.Dibco2010() + Dibco.Dibco2011() + Dibco.Dibco2012() + Dibco.Dibco2013() + Dibco.Dibco2014() + Dibco.Dibco2016() + Dibco.Dibco2017() + Dibco.Dibco2018()
        validationset = Dibco.Dibco2019()
    train_resizer= tormentor.RandomSaturation.custom(saturation=Constant(0.0)) & tormentor.PadCropTo.new_size(p.patch_width, p.patch_height)
    val_resizer= tormentor.RandomSaturation.custom(saturation=Constant(0.0)) & tormentor.PadCropTo.new_size(eval(p.val_patch_width), eval(p.val_patch_height))
    trainset = tormentor.AugmentedDs(trainset, augmentation_factory=train_resizer, computation_device="cpu", add_mask=True)
    validationset = tormentor.AugmentedDs(validationset, augmentation_factory=val_resizer, computation_device="cpu", add_mask=True)
    trainloader=torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=p.batch_size, num_workers= p.io_threads, drop_last=True)
    valloader=torch.utils.data.DataLoader(validationset, shuffle=False, batch_size=1)
    if p.train_augmentation != "":
        augmentation_factory = tormentor.AugmentationFactory(eval(p.train_augmentation))
        trainloader = tormentor.AugmentedDataLoader(trainloader, augmentation_factory, torch.device(p.tormentor_device), augment_RRDS_batch)
    if p.val_augmentation != "":
        augmentation_factory = tormentor.AugmentationFactory(eval(p.val_augmentation))
        valloader = tormentor.AugmentedDataLoader(valloader, augmentation_factory, torch.device(p.tormentor_device), augment_RRDS_batch)
else:
    raise ValueError()


net = create_net(p.arch, p.n_channels, p.n_classes, p.bn_momentum, p.rnd_pad, p.pretrained)
param_hist, per_epoch_train_errors, per_epoch_validation_errors, start_epoch, net = resume(net, p.resume_fname, p.device)


optim = torch.optim.Adam(net.parameters(), lr=p.lr)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')


for epoch in range(start_epoch, p.epochs):
    print(f"Epoch {epoch}")
    if p.save_freq != 0 and epoch % p.save_freq==0:
        param_hist[epoch] = param_dict
        save(param_hist, per_epoch_train_errors, per_epoch_validation_errors,epoch,net)
    if p.validate_freq != 0 and epoch % p.validate_freq == 0:
        fscore,precision,recall, loss=run_epoch(p,p.val_device, valloader, net, criterion, optimizer=None, save_images=p.save_images, is_deeplabv3=False,save_input_images=p.save_input_images)
        per_epoch_validation_errors[epoch]=fscore,precision,recall,loss
    save_outputs=p.trainoutputs_freq != 0 and epoch % p.trainoutputs_freq == 0
    fscore, precision, recall, loss=run_epoch(p,p.device, trainloader, net, criterion, optimizer=optim, save_images=p.save_images, is_deeplabv3=False, save_input_images=p.save_input_images)
    per_epoch_train_errors[epoch]=fscore, precision, recall, loss