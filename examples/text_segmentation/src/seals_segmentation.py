import glob
import re
from PIL import Image
from dibco import dibco_transform_gray_input
import torch
import json
import torchvision


def random_multicrop_to(dst_wh, *tensors):
    assert len(tensors) > 0
    assert len(tensors[0].size()) == 3
    src_w = tensors[0].size(2)
    src_h = tensors[0].size(1)
    dst_w, dst_h = dst_wh

    if src_w <= dst_w:
        left = 0 
        right = src_w
    else:
        left = torch.randint(0, src_w - dst_w, (1, 1)).item()
        right = left + dst_w
    if src_h <= dst_h:
        top = 0
        bottom = src_h
    else:
        top = torch.randint(0, src_h - dst_h, (1, 1)).item()
        bottom = top + dst_h

    res = []
    for t in tensors:
        assert len(t.size()) == 3
        assert t.size(2) >= src_w
        assert t.size(1) >= src_h
        res.append(t[:, top:bottom, left:right])
    return res


def paint_boxes(res_sz, gt_dict, draw_class_list, erase_class_list, bg=1):
    draw_classes = [gt_dict["class_names"].index(c) for c in draw_class_list]
    erase_classes = [gt_dict["class_names"].index(c) for c in erase_class_list]
    draw_boxes = [gt_dict["rect_LTRB"][i] for i in range(len(gt_dict["rect_classes"])) if gt_dict["rect_classes"][i] in draw_classes]
    erase_boxes = [gt_dict["rect_LTRB"][i] for i in range(len(gt_dict["rect_classes"])) if gt_dict["rect_classes"][i] in erase_classes]
    res_img = torch.empty(res_sz)
    res_img.fill_(bg)

    for l, t, r, b in erase_boxes:
        res_img[:, t:b, l:r] = 0
    for l, t, b, r in draw_boxes:
        res_img[:, t:b, l:r] = 1
    return res_img


class SealDs:
    def crop_sample(self, img, gt, mask):
        if self.desired_width is None and self.desired_height is None:
            return img, gt, mask
        else:
            img, gt, mask = random_multicrop_to((self.desired_width, self.desired_height), img, gt, mask)
            return img, gt, mask
    
    def __init__(self, draw_class_list=[], erase_class_list=["Wr:OldText"], file_glob_list=["../../../ptlbp/data/1000CV/1000_CVCharters/*/*/*/*.seals.gt.json"], extention_regex=".seals.gt.json", img_replace_glob=".img.*", input_transform=dibco_transform_gray_input, desired_width=None, desired_height=None):
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.input_transform = input_transform
        all_gt_files = []
        for file_glob in file_glob_list:
            all_gt_files += list(glob.glob(file_glob, recursive=True))
        self.erase_class_list = erase_class_list
        self.draw_class_list = draw_class_list
        self.img_files = []
        self.gt_files = []
        for gt_file in all_gt_files:
            img_glob = re.sub(extention_regex, img_replace_glob, gt_file)
            imgs = glob.glob(img_glob, recursive=True)
            #  print(f"img_glob: {img_glob}")
            if len(imgs) == 1:
                self.gt_files.append(gt_file)
                self.img_files.append(imgs[0])
            else:
                #  print(f"img_glob: {img_glob} {len(imgs)}")
                assert len(imgs) == 0, f"Found more than one image for {gt_file}"

    def __getitem__(self, n):
        input_img = self.input_transform(Image.open(self.img_files[n]))
        gt = torch.empty_like(input_img)[[0, 0], :, :]
        gt[0, :, :] = 0
        gt[1, :, :] = 1
        gt_dict = json.load(open(self.gt_files[n]))
        mask = paint_boxes(gt.size(), gt_dict, self.draw_class_list, self.erase_class_list)
        return self.crop_sample(input_img, gt, mask)

    def __len__(self):
        return len(self.img_files)