#!/usr/bin/env python3
import timeit

#pip3 install --user fargv # You won't regret it ;-)
import fargv

args = {
    "dataset_size": 1000,
    "repeat": 1,
    "batch_size": 100,
    "num_workers": 1,
    "image_width": 224,
    "image_height": 224,
    "image_channels": 3,
    "device": "cpu",
    "augmentation": ("Rotate", "Flip"),
    "forward": (False,
                "Should a forward pass be added to estimate the overall effect of the augmentation to the training."
                "Image size must be 3x224x224 if this is enabled"),
    "backward": (False,
                 "Should a backward pass be added to estimate the overall effect of the augmentation to the training,"
                 " if forward is false, this is ignored")
}

args, _ = fargv.fargv(args)

if args.augmentation == "Rotate":
    kornia_augmentation = "kornia.augmentation.RandomHorizontalFlip"
elif args.augmentation == "Flip":
    kornia_augmentation = "kornia.augmentation.RandomRotation"


def bench_kornia_batch(args):
    img_sizes = [40, 200, 1000]
    batch_sizes = [1, 3, 10, 30, 100]
    devices = ["cpu"]  # , "cuda"]
    by_sample = {}
    by_batch = {}
    for img_size in img_sizes:
        for batch_size in batch_sizes:
            for device in devices:
                setup = f"""import torch, kornia
    img = torch.rand([{batch_size}, 1, {img_size}, {img_size}]).to('{device}')
    degres = torch.rand({batch_size}).to('{device}')*360
    """
                per_batch = "kornia.geometry.rotate(img,rotate_degrees)"
                per_sample = f"torch.cat([kornia.geometry.rotate(img[:1,:,:,:],rotate_degrees[:1]) for _ in range({batch_size}) ],dim=0)"
                by_batch[(img_size, batch_size, device)] = timeit.timeit(setup=setup, stmt=per_batch,
                                                                         number=args.repeat)
                by_sample[(img_size, batch_size, device)] = timeit.timeit(setup=setup, stmt=per_sample,
                                                                          number=args.repeat)
    return by_batch, by_sample


setup_str = f"""
import torchvision
import torch
import tormentor
import kornia


# About issue https://discuss.pytorch.org/t/not-using-multiprocessing-but-getting-cuda-error-re-forked-subprocess/54610/7
#torch.multiprocessing.set_start_method('spawn')
torch.multiprocessing.set_start_method('forkserver', force=True)


dataset=[(torch.rand(1, {args.image_channels}, {args.image_width}, {args.image_height}), 0) for _ in range({args.dataset_size})]

# Tormentor per sample augmentation
augmentation_factory = tormentor.{args.augmentation}.factory()
augmented_dataset = tormentor.AugmentationDataset(dataset, augmentation_factory)
per_sample_loader = torch.utils.data.DataLoader(augmented_dataset,num_workers={args.num_workers},batch_size={args.batch_size})

# Minibatch augmentation
per_batch_unaugmented_loader = torch.utils.data.DataLoader(dataset, num_workers={args.num_workers},batch_size={args.batch_size})

def per_batch_loader():
    for batch_img, batch_labels in per_batch_unaugmented_loader:
        #print(".")
        print(batch_img.size(), end='')
        batch_img={kornia_augmentation}()(batch_img[:,0,:,:,:])
        print(batch_img.size())
        yield (batch_img, batch_labels) 

net = torchvision.models.vgg13().to({repr(args.device)})
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def run_epoch(by_batch):
    if by_batch:
        data_loader = per_batch_loader()
    else:
        data_loader = per_sample_loader
    for inputs, targets in data_loader:
        inputs, targets = inputs.to({repr(args.device)}), targets.to({repr(args.device)})
        if {repr(args.forward)}:
            output=net(inputs)
            loss=criterion(output,targets)
            if {repr(args.backward)}:
                loss.sum().backward()
                optimizer.step()
                optimizer.zero_grad()
print("Setup Complete")
"""

if __name__ == "__main__":
    for n,line in enumerate(setup_str.split("\n")):
        print(n,":",line)
    by_batch = timeit.timeit(setup=setup_str, stmt="run_epoch(by_batch=True)", number=args.repeat)
    by_sample = timeit.timeit(setup=setup_str, stmt="run_epoch(by_batch=False)", number=args.repeat)
    print(f"By Batch duration {by_batch} sec.")
    print(f"By Sample duration {by_sample} sec.")
