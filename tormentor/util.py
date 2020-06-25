import torch
import PIL
import kornia as K
from PIL import Image, ImageDraw,ImageFont
import numpy as np


def render_text(text,font=None,fnt_path="", pad=10, font_size=32):
    if font is None:
        font=ImageFont.truetype(fnt_path, font_size)
    else:
        assert fnt_path is ""
    ascent, descent = font.getmetrics()
    (width, baseline), _ = font.font.getsize(text)
    blank_image = Image.new('RGBA', (width+2*pad, ascent+descent+2*pad), 'white')
    img_draw = ImageDraw.Draw(blank_image)
    img_draw.text((pad, pad), text, fill='black',font=font)
    return K.utils.image_to_tensor(np.array(blank_image))/255.0


def load_images_as_tensors(image_filename_list, channel_count, min_width_height, preserve_aspect_ratio=True):
    assert channel_count in (1, 2, 3, 4)
    for image_file_name in image_filename_list:
        img = PIL.Image.open(image_file_name)
        if channel_count == 1:
            img = img.convert("L")
        elif channel_count == 2:
            img = img.convert("LA")
        elif channel_count == 3:
            img = img.convert("RGB")
        else:  # channel_count == 4
            img = img.convert("RGBA")
        width, height = img.size
        if min_width_height is not None and (width < min_width_height[0] or height < min_width_height[1]):
            if preserve_aspect_ratio:
                width, height = img
                width_scale = min_width_height[0] / width
                height_scale = min_width_height[1] / height
                scale = max(width_scale, height_scale)
                img = img.resize((int(scale * width), int(scale * height)))
            else:
                img = img.resize(min_width_height)
        yield K.image_to_tensor(img).unsqueeze(dim=0)


def debug_pattern(num_rows:int, num_cols:int,square_size:int)->torch.FloatTensor:
    res_img = torch.zeros([num_cols*square_size,num_rows*square_size])
    square = (torch.arange(square_size).view(1,square_size) * torch.arange(square_size).view(square_size,1))
    square = square / float(square_size * square_size)
    for y in range(0,num_rows):
        for x in range(y%2, num_cols,2):
            res_img[x*square_size:(x+1)*square_size, y*square_size:(y+1)*square_size] = square
    return res_img.unsqueeze(dim=0).unsqueeze(dim=0)


def PCA(X, k, center=True, scale=False):
    n, p = X.size()
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    X_center = torch.mm(H.double(), X.double())
    covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
    scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
    eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
    components = (eigenvectors[:, :k]).t()
    explained_variance = eigenvalues[:k, 0]
    return {'X': X, 'k': k, 'components': components, 'explained_variance': explained_variance}
