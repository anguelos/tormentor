#!/usr/bin/env python3
import fargv
import torchvision
import tormentor
import PIL
import json

def load_as_batch(image_fname, replicates):
    img = torchvision.transforms.ToTensor()(PIL.Image.open(image_fname))
    batch = img.unsqueeze(dim=0).repeat([replicates, 1, 1, 1])
    return batch

def save_batch(batch_tensor, image_fname, size, nrow):
    image_tensor = torchvision.utils.make_grid(batch_tensor, nrow=nrow).detach().cpu()
    return torchvision.transforms.ToPILImage()(image_tensor).resize(size).save(image_fname)


p = {
    "augmentation_list_path": "./augmentation_example_list.json",
    "input_image": "./source/_static/img/test_card.png",
    "output_path": "./source/_static/example_images/",
    "replicates": 10,
    "nrow": 5,
    "out_width": 500,
    "out_height": 200,
    "device": "cpu"
}

p, _ = fargv.fargv(p, return_named_tuple=True)

input_batch = load_as_batch(p.input_image, p.replicates)
input_batch = input_batch.to(p.device)
for augmentation_name, augmentation_cls_str in json.load(open(p.augmentation_list_path)).items():
    print("Computing ", augmentation_name, end="",flush=True)
    out_fname = f"{p.output_path}/{augmentation_name}.png"


    augmentation = eval(augmentation_cls_str)()
    output_batch = augmentation(input_batch)
    output_batch[0, :, :, :] = input_batch[0, :, :, :]

    save_batch(output_batch, out_fname, (p.out_width, p.out_height), p.nrow)
    print(": finished.",flush=True)
