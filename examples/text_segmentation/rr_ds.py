import io
import zipfile
import PIL
import re
import torchvision
import torch
from collections import defaultdict
import string
import numpy as np

def return_one():
    return 1

def re_int(x):
    return int(re.findall("[0-9]+",x)[0])

input_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
def img2gt(x):
    return (x<.99).float()
gt_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor(), img2gt])


def create_alphabet():
    symbol_map = defaultdict(return_one)
    symbol_map[" "] = 0
    base = max((symbol_map.values())) + 1
    symbol_map.update({c: n + base for n, c in enumerate(string.ascii_lowercase)})
    symbol_map.update({c: n + base for n, c in enumerate(string.ascii_uppercase)})
    base = max((symbol_map.values())) + 1
    symbol_map.update({c: n + base for n, c in enumerate(string.digits)})
    return symbol_map


class RR2013Ch2():
    # @staticmethod
    # def create_alphabet():
    #     symbol_map = defaultdict(lambda: 1)
    #     symbol_map[" "] = 0
    #     base = max((symbol_map.values())) + 1
    #     symbol_map.update({c: n + base for n, c in enumerate(string.ascii_lowercase)})
    #     symbol_map.update({c: n + base for n, c in enumerate(string.ascii_uppercase)})
    #     base = max((symbol_map.values())) + 1
    #     symbol_map.update({c: n + base for n, c in enumerate(string.digits)})
    #     return symbol_map

    def __init__(self, train=True, return_char_gt=False, return_mask=True,
                 reduce_img_size=True,
                 default_width=512+128,
                 default_height=512+128,
                 cache_ds=True,
                 input_transform=input_transform,
                 gt_transform=gt_transform,
                 root="/home/anguelos/data/rr/focused_segmentation/zips",
                 train_input_fname="Challenge2_Training_Task12_Images.zip",
                 train_gt_fname="Challenge2_Training_Task2_GT.zip",
                 test_input_fname="Challenge2_Test_Task12_Images.zip",
                 test_gt_fname="Challenge2_Test_Task2_GT.zip"):
        self.train = train
        self.reduce_img_size = reduce_img_size
        if train:
            input_fname = train_input_fname
            gt_fname = train_gt_fname
        else:
            input_fname=test_input_fname
            gt_fname=test_gt_fname

        #self.char2int = RR2013Ch2.create_alphabet()
        self.char2int = create_alphabet()
        self.max_class = max(self.char2int.values())
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.input_zipfile=zipfile.ZipFile(f"{root}/{input_fname}")
        self.gt_zipfile=zipfile.ZipFile(f"{root}/{gt_fname}")
        self.id2input_img={re_int(f.filename):f for f in self.input_zipfile.filelist}
        self.id2gt_img={re_int(f.filename):f for f in self.gt_zipfile.filelist if f.filename[-3:] in ["png", "bmp", "jpg"]}
        self.id2gt_txt={re_int(f.filename):f for f in self.gt_zipfile.filelist if f.filename[-3:] =="txt"}
        self.ids = sorted(self.id2gt_img.keys())
        self.return_char_gt = return_char_gt
        self.return_mask = return_mask
        self.train = train
        self.default_width=default_width
        self.default_height=default_height
        self.cache_ds=cache_ds
        if self.cache_ds:
            try:
                print("loading...")
                self.cache=torch.load(f"/tmp/ds_train_{repr(self.train)}.pt")
                print("success!!!")
            except:
                self.cache={}
                for n in range(len(self)):
                    print(".", end="")
                    _ = self[n]
                print("saving...")
                torch.save(self.cache, f"/tmp/ds_train_{repr(self.train)}.pt")
                print("success!!!")
            del self.input_zipfile
            del self.gt_zipfile


        assert self.ids == sorted(self.id2gt_img.keys())


    def read_txt(self, id, gt_img):
        txt = self.gt_zipfile.read(self.id2gt_txt[id]).decode("utf-8")
        lines = [l.strip() for l in txt.strip().replace('"""','"\\""').split("\n") if len(l.strip()) > 0]
        cares = [l for l in lines if l[0] != "#"]
        parse_re = re.compile('[0-9]+|\\".+\\"')
        cares = [[eval(c) for c in parse_re.findall(l)] for l in cares]
        if any([len(c)!=10 for c in cares if len(c)!=10]):
            print("WARNING:!!!!!!")
            [print(l) for l in lines if l[0] != "#"]
            [print(l) for l in lines if l[0] != "#"]
            print(self.id2gt_txt[id])
            print()
        cares = [c for c in cares if len(c)==10]

        dont_cares = [l for l in lines if l[0] == "#"]
        dont_cares = [[eval(c) for c in parse_re.findall(l)] for l in dont_cares]
        gt_rgb=np.asarray(gt_img).transpose([2,1,0])
        symbol_img = torch.zeros([self.max_class+1, gt_rgb.shape[1], gt_rgb.shape[2]])
        dont_care_img = torch.zeros_like(symbol_img[0,:,:])
        for r, g, b, cx, cy, left, top, right, bottom, symbol in cares:
            char_id = r+g*255+b*255*256
            region_mask=(gt_rgb[0, left:right+1, top:bottom+1]==r)*(gt_rgb[1, left:right+1, top:bottom+1]==g)*(gt_rgb[2, left:right+1, top:bottom+1]==b)
            symbol_img[self.char2int[symbol],left:right+1, top:bottom+1]=(torch.tensor(region_mask)).float()

        for r, g, b, cx, cy, left, top, right, bottom, symbol in dont_cares:
            char_id = r+g*255+b*255*256
            region_mask=(gt_rgb[0, left:right+1, top:bottom+1]==r)*(gt_rgb[1, left:right+1, top:bottom+1]==g)*(gt_rgb[2, left:right+1, top:bottom+1]==b)
            dont_care_img[left:right+1, top:bottom+1]=torch.tensor(region_mask).float()

        symbol_img = symbol_img.transpose(1,2)
        dont_care_img = dont_care_img.transpose(1,0)
        symbol_img[0,:,:] = 1-(symbol_img[1:,:,:].sum(dim=0))
        return symbol_img, (1-dont_care_img).unsqueeze(dim=0)


    def construct_item(self, item):
        id = self.ids[item]
        in_img_blob = self.input_zipfile.read(self.id2input_img[id])
        gt_img_blob = self.gt_zipfile.read(self.id2gt_img[id])
        in_img = PIL.Image.open(io.BytesIO(in_img_blob))
        gt_img = PIL.Image.open(io.BytesIO(gt_img_blob))
        in_img = self.input_transform(in_img)
        if (self.return_mask or self.return_char_gt) and self.train:
            symbol_img, mask = self.read_txt(id, gt_img)
            if not self.return_char_gt:
                #    gt_img = torch.cat([1-symbol_img[:1,:,:], symbol_img[:1,:,:]], dim=0)
                gt_img = symbol_img[1:, :, :].sum(dim=0).unsqueeze(dim=0)
            else:
                gt_img = symbol_img
        else:
            gt_img = self.gt_transform(gt_img)
            mask=torch.ones_like(gt_img[:1,:,:])
        results=[in_img, gt_img]
        if self.return_mask:
            results.append(mask)
        return results

    def reduce_size(self, sample_list):
        old_width = sample_list[0].size(-1)
        old_height = sample_list[0].size(-2)

        new_width = self.default_width
        new_height = self.default_height

        if old_height < new_height:
            v_pad = (1+new_height-old_height)//2
            sample_list=[torch.nn.functional.pad(input=s.unsqueeze(dim=0), pad=(0, 0, v_pad, v_pad), mode='constant', value=1)[0,:,:,:] for s in sample_list]
            old_height = sample_list[0].size(-2)
        if old_width < new_width :
            h_pad = (1+new_width-old_width)//2
            sample_list=[torch.nn.functional.pad(input=s.unsqueeze(dim=0), pad=(h_pad, h_pad, 0, 0), mode='constant', value=1)[0,:,:,:] for s in sample_list]
            old_width = sample_list[0].size(-1)
        if not self.reduce_img_size:
            return sample_list
        left = np.random.randint(0, 1+old_width-new_width)
        top = np.random.randint(0, 1+old_height-new_height)
        results = [r[:, top:top+new_height,left:left+new_width] for r in sample_list]
        return results

    def __getitem__(self, item):
        if self.cache_ds:
            results=self.cache.get(item, None)
            if results is None:
                results=self.construct_item(item)
                self.cache[item]=results
        else:
            results=self.construct_item(item)
        if results[1].size(0) == 1 and results[1].dtype==torch.float32:
            results[1] = torch.cat([1-results[1],results[1]],dim=0)
        return self.reduce_size(results)

    def __len__(self):
        return len(self.ids)
