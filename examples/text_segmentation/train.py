#!/usr/bin/env python3
print("Importing fastai")
import fastai
print("Importing fastai.vision")
import fastai.vision
print("done")

import torch
import fargv

from unet import UNet
from rr_ds import RR2013Ch2
#from dagtasets import Dibco
import lm_util
import sys
import cv2
import time
import numpy as np

#from matplotlib import pyplot as plt

p={
    "io_threads": 1,
    "log_freq": 10,
    "lr": .001,
    "epochs": 10,
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


def render_confusion(prediction,gt,tp_col=[0, 0, 0],tn_col=[255,255,255],fp_col=[255,0,0],fn_col=[0,0,255]):
    prediction=(prediction.cpu().numpy())
    gt = (gt.cpu().numpy())
    res=np.zeros(prediction.shape+(3,))
    tp = gt & prediction
    tn = (~gt) & (~prediction)
    fp = (~gt) & prediction
    fn = (gt) & (~prediction)
    res[tp, :] = tp_col
    res[tn, :] = tn_col
    res[fp, :] = fp_col
    res[fn, :] = fn_col
    precision = (1+tp.sum())/float(1+tp.sum()+fp.sum())
    recall = (1+tp.sum()) / float(1+tp.sum() + fn.sum())
    Fscore=(2*precision*recall)/(precision+recall)
    return res,precision,recall,Fscore


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
            coeffs = coeffs * mask
            #coeffs=1
            if p.mode == "normal":
                if is_deeplabv3:
                    #print(input_img.size())
                    out_dict = net(input_img)
                    prediction = out_dict["out"]
                    loss_mask = out_dict["aux"]
                    loss = criterion(prediction, gt)*coeffs * loss_mask
                    prediction = torch.nn.functional.softmax(prediction, dim=1)
                    prediction = torch.cat([prediction,1-prediction],dim=1).detach()
                else:
                    prediction = net(input_img)
                    loss = criterion(prediction, gt)* coeffs
                    prediction = torch.nn.functional.softmax(prediction, dim=1)
                    prediction = torch.cat([prediction,1-prediction],dim=1).detach()

            # elif p.mode == "residual":
            #     prediction = net(input_img) + torch.log(input_img)
            #     loss = criterion(prediction, gt)*coeffs
            #     prediction = torch.nn.functional.softmax(prediction, dim=1)
            #     prediction = torch.cat([prediction,1-prediction],dim=1).detach()
            #
            # elif p.mode == "chain":
            #     prediction = net(input_img)
            #     #loss = criterion(prediction, gt)*coeffs
            #     prediction = torch.nn.functional.softmax(prediction, dim=1)
            #     prediction = torch.cat([prediction[:,:1,:,:],1-prediction[:,:1,:,:]],dim=1)
            #
            #     prediction = net(prediction)
            #     #loss = loss + criterion(prediction, gt).sum()*coeffs
            #     loss = criterion(prediction, gt) * coeffs
            #     prediction = torch.nn.functional.softmax(prediction, dim=1)
            #     prediction = torch.cat([prediction,1-prediction],dim=1).detach()
            #
            # elif p.mode == "residual_chain":
            #     prediction = net(input_img) + torch.log(input_img)
            #     #loss = criterion(prediction, gt)*coeffs
            #     prediction = torch.nn.functional.softmax(prediction, dim=1)
            #     prediction = torch.cat([prediction[:,:1,:,:],1-prediction[:,:1,:,:]],dim=1)
            #
            #     prediction = net(prediction) + torch.log(prediction)
            #     #loss = loss + criterion(prediction, gt)*coeffs
            #     loss = criterion(prediction, gt) * coeffs
            #
            #     prediction = torch.nn.functional.softmax(prediction, dim=1)
            #     prediction = torch.cat([prediction,1-prediction],dim=1).detach()
            else:
                raise ValueError("unknown mode")
            loss=loss.sum()
            if not is_validation:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            #bin_prediction = prediction[0,0,:,:]<get_otsu_threshold(prediction[0,0,:,:])
            #confusion,precision,recall,fscore = render_confusion(bin_prediction, gt[0, 0, :, :]<.5)
            confusion, precision, recall, fscore = render_confusion(prediction[0,0,:,:]<.5, gt[0, 0, :, :] < .5)
            #confusion,precision,recall,fscore = render_optimal_confusion(prediction[0,0,:,:], gt[0, 0, :, :]<.5)

            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)
            losses.append(loss.item()/gt.view(-1).size()[0])
            if save_images:
                cv2.imwrite("/tmp/{}_{}_img_{}.png".format(p.mode,isval_str,n), confusion)
                model_outputs[n]=prediction.detach().cpu()
    #lines = ["{} {}:\t{}".format(isval_str[0],n,fscores[n]) for n in range(len(fscores))]
    lines = []
    lines.append("Epoch {} {} Total:\t{:05f}%".format(epoch, isval_str, 100*sum(fscores)/len(fscores)))
    lines.append('')
    print("N:\t{} {} % computed in {:05f} sec.".format(p.mode,isval_str,time.time()-t))
    print("\n".join(lines))
    if save_images:
        torch.save(model_outputs,"/tmp/{}_{}_samples.pt".format(p.mode,isval_str))
    return sum(fscores) / len(fscores),sum(precisions) / len(precisions), sum(recalls) / len(recalls),sum(losses) / len(losses)



trainset = RR2013Ch2(train=True, return_mask=True)
testset = RR2013Ch2(train=False, return_mask=True, cache_ds=False)

device = torch.device(p.device)

print("Loading Data into loaders... ", end="")
trainloader=torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=p.batch_size, num_workers= p.io_threads, drop_last=True)
valloader=torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1)
print("done!")

net = UNet(n_channels=trainset[0][0].size(0), n_classes=trainset[0][1].size(0))


print("Creating Resnet")
body = fastai.vision.learner.create_body(fastai.vision.models.resnet34, pretrained=True, cut=-2)
print("Creating Unet")
net = fastai.vision.models.unet.DynamicUnet(body, trainset[0][1].size(0))
print("Done")
#net = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
#net.eval()
#net.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
#net.aux_classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

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