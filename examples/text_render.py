#from .base_augmentation import SpatialImageAugmentation
import tormentor
from matplotlib import pyplot as plt
import numpy as np
import kornia as K
from PIL import Image, ImageDraw,ImageFont
from tormentor import WrapAugmentation



def render_text(text,font=None,fnt_path="", pad=10, font_size=32):
    if font is None:
        font=ImageFont.truetype(fnt_path, font_size)
    else:
        assert fnt_path is ""
    ascent, descent = font.getmetrics()
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    blank_image = Image.new('RGBA', (width+2*pad, ascent+descent+2*pad), 'white')
    img_draw = ImageDraw.Draw(blank_image)
    img_draw.text((pad, pad), text, fill='black',font=font)
    return K.utils.image_to_tensor(np.array(blank_image))/255.0


text='Hello World asdasf'*3
fnt_path="/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
img=render_text(text,fnt_path=fnt_path).unsqueeze(dim=0)
f,ax_list = plt.subplots(2,1)

aug=tormentor.WrapAugmentation()
print(img.size())
ax_list[0].imshow(img[0,0,:,:],cmap="gray")
ax_list[1].imshow(aug(img)[0,0,:,:],cmap="gray")
plt.show()
