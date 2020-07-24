#!/usr/bin/env python3
import fastai.vision
import tormentor

import torch
import fargv


from rr_ds import RR2013Ch2
from dagtasets import Dibco
import cv2
import time
import numpy as np

#from matplotlib import pyplot as plt

p={
    "io_threads": 1,
    "log_freq": 10,
    "lr": .001,
    "epochs": 10,
    "tormentor_device":"cpu",
    "device":"cuda",
    "val_device":"{device}",
    "validate_freq": 5,
    "trainoutputs_freq": 5,
    "batch_size": 1,
    "save_freq": 10,
    "mask_gt": 1,
    "resume_fname":"binnet_{mode}.pt",
    "mode": ("normal",'One of ["normal","residual","chain","residual_chain"].')
}

p, _ = fargv.fargv(p, return_named_tuple=True)


def save(per_epoch_train_errors,per_epoch_validation_errors,epoch,net):
    save_dict=net.state_dict()
    save_dict["per_epoch_train_errors"]=per_epoch_train_errors
    save_dict["per_epoch_validation_errors"] = per_epoch_validation_errors
    save_dict["epoch"]=epoch
    torch.save(save_dict, p.resume_fname)

def resume(net):
    try:
        save_dict=torch.load(p.resume_fname)
        per_epoch_train_errors=save_dict["per_epoch_train_errors"]
        del save_dict["per_epoch_train_errors"]
        per_epoch_validation_errors=save_dict["per_epoch_validation_errors"]
        del save_dict["per_epoch_validation_errors"]
        start_epoch=save_dict["epoch"]
        del save_dict["epoch"]
        net.load_state_dict(save_dict)
        print("Resumed from ",p.resume_fname)
        return per_epoch_train_errors,per_epoch_validation_errors,start_epoch,net
    except FileNotFoundError as e:
        print("Failed to resume from ", p.resume_fname)
        return {},{},0, net


def render_confusion(prediction,gt,valid_mask=None,tp_col=[0, 0, 0],tn_col=[255,255,255],fp_col=[255,0,0],fn_col=[0,0,255], undetermined_col=[128,128,128]):
    prediction=(prediction.cpu().numpy())
    gt = (gt.cpu().numpy())
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


def run_epoch(p,device,loader,net,criterion,optimizer=None,save_images=True,is_deeplabv3=True):
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
        for n, (input_img,gt,mask) in enumerate(loader):
            input_img, gt, mask = input_img.to(device), gt.to(device), mask.to(device)
            coeffs = gt[:, 1, :, :].mean(dim=1).mean(dim=1)
            coeffs = coeffs.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            #print("coefs.size()",coeffs.size(),"  mask.size()", mask.size())
            coeffs = coeffs * mask
            #coeffs=1
            if p.mode == "normal":
                if is_deeplabv3:
                    #print(input_img.size())
                    out_dict = net(input_img)
                    prediction = out_dict["out"]
                    loss_mask = out_dict["aux"]
                    loss = criterion(prediction, gt)
                    loss = loss*coeffs * loss_mask
                    prediction = torch.nn.functional.softmax(prediction, dim=1)
                    prediction = torch.cat([prediction,1-prediction],dim=1).detach()
                else:
                    prediction = net(input_img)
                    #print("Predictions",prediction.size(),gt.size())
                    loss = criterion(prediction, gt)
                    #print("Losses",loss.size(),coeffs.size())
                    loss = loss* coeffs
                    prediction = torch.nn.functional.softmax(prediction, dim=1)
                    prediction = torch.cat([prediction,1-prediction],dim=1).detach()

            else:
                raise ValueError("unknown mode")
            loss=loss.sum()
            if not is_validation:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            #bin_prediction = prediction[0,0,:,:]<get_otsu_threshold(prediction[0,0,:,:])
            #confusion,precision,recall,fscore = render_confusion(bin_prediction, gt[0, 0, :, :]<.5)
            confusion, precision, recall, fscore = render_confusion(prediction[0,0,:,:]<.5, gt[0, 0, :, :] < .5, mask[0,0,:,:] > .5)
            #confusion,precision,recall,fscore = render_optimal_confusion(prediction[0,0,:,:], gt[0, 0, :, :]<.5)

            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)
            losses.append(loss.item()/gt.view(-1).size()[0])
            if save_images:
                cv2.imwrite("/tmp/{}_{}_img_{}.png".format(p.mode,isval_str,n), confusion)
                model_outputs[n]=prediction.detach().cpu()
    lines = []
    lines.append("Epoch {} {} Total:\t{:05f}%".format(epoch, isval_str, 100*sum(fscores)/len(fscores)))
    lines.append('')
    print("N:\t{} {} % computed in {:05f} sec.".format(p.mode,isval_str,time.time()-t))
    print("\n".join(lines))
    if save_images:
        torch.save(model_outputs,"/tmp/{}_{}_samples.pt".format(p.mode,isval_str))
    return sum(fscores) / len(fscores),sum(precisions) / len(precisions), sum(recalls) / len(recalls),sum(losses) / len(losses)



augmentation_cls=tormentor.cascade([tormentor.Wrap,tormentor.Perspective])

trainset = RR2013Ch2(train=True, return_mask=True,cache_ds=True)


def augment_RRDS_batch(batch, augmentation,process_device):
    input_imgs, segmentations, masks = batch
    if process_device != batch[0].device:
        input_imgs=input_imgs.to(process_device)
        segmentations=segmentations.to(process_device)
        masks=masks.to(process_device)
    with torch.no_grad():
        input_imgs = augmentation(input_imgs)
        segmentations = augmentation(segmentations, is_mask=True)
        masks = augmentation(masks, is_mask=True)
        segmentations = torch.clamp(segmentations[:,:1, :, :] + (1-masks),0.,1.0)
        segmentations = torch.cat([segmentations, 1-segmentations], dim=1)
        if process_device != batch[0].device:
            return input_imgs.to(batch[0].device), segmentations.to(batch[0].device), masks.to(batch[0].device)
        else:
            return input_imgs, segmentations, masks


def augment_RRDS_sample(sample, augmentation,process_device):
    input_img, segmentation, mask = sample
    if process_device != sample[0].device:
        input_img=input_img.to(process_device)
        segmentation=segmentation.to(process_device)
        mask = mask.to(process_device)
    with torch.no_grad():
        input_img = augmentation(input_img)
        segmentation = augmentation(segmentation, is_mask=True)
        mask = augmentation(mask, is_mask=True)
        segmentation = torch.clamp(segmentation[:1, :, :] + (1-mask),0.,1.0)
        segmentation = torch.cat([segmentation, 1-segmentation], dim=0)
        if process_device != sample[0].device:
            return input_img.to(sample[0].device), segmentation.to(sample[0].device), mask.to(sample[0].device)
        else:
            return input_img, segmentation, mask






#trainset=WrapRRDS(trainset,augmentation_cls,torch.device(p.tormentor_device))

img,segm,mask = trainset[0]



testset = RR2013Ch2(train=False, return_mask=True, cache_ds=True)


device = torch.device(p.device)

print("Loading Data into loaders... ", end="")
trainloader=torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=p.batch_size, num_workers= p.io_threads, drop_last=True)

#trainloader=WrapRRDL(trainloader,augmentation_cls,torch.device(p.tormentor_device))
trainloader = tormentor.AugmentedDataLoader(trainloader, augmentation_cls, torch.device(p.tormentor_device), augment_RRDS_batch)
valloader=torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1)
print("done!")

#net = UNet(n_channels=trainset[0][0].size(0), n_classes=trainset[0][1].size(0))


print("Creating Resnet")
body = fastai.vision.learner.create_body(fastai.vision.models.resnet34, pretrained=True, cut=-2)
print("Creating Unet")
net = fastai.vision.models.unet.DynamicUnet(body, trainset[0][1].size(0))
print("Done")

print("Resuming Model ... ", end="")
per_epoch_train_errors,per_epoch_validation_errors,start_epoch,net=resume(net)
print("done!")

optim = torch.optim.Adam(net.parameters(), lr=p.lr)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')


for epoch in range(start_epoch, p.epochs):
    if p.save_freq != 0 and epoch % p.save_freq==0:
        save(per_epoch_train_errors,per_epoch_validation_errors,epoch,net)
    if p.validate_freq != 0 and epoch % p.validate_freq == 0:
        fscore,precision,recall,loss=run_epoch(p,p.val_device, valloader, net, criterion, optimizer=None, save_images=True, is_deeplabv3=False)
        per_epoch_validation_errors[epoch]=fscore,precision,recall,loss
    save_outputs=p.trainoutputs_freq!=0 and epoch % p.trainoutputs_freq==0
    fscore, precision, recall, loss=run_epoch(p,p.device, trainloader, net, criterion, optimizer=optim, save_images=save_outputs, is_deeplabv3=False)
    per_epoch_train_errors[epoch]=fscore, precision, recall, loss