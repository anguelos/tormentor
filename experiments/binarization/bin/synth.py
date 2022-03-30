#!/usr/bin/env python3
import rpack
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import glob
import fargv
import tqdm
import pathlib


p = {
    "corpus":"./data/brown.txt",
    "outdir":"./tmp/synth",
    "fontdir":"./data/fonts",
    "npages":1000
}

p, _ = fargv.fargv(p)

lines = open(p.corpus).readlines()
lines = [l for l in lines if len(l.strip())>0 and not l.strip().startswith("#")]
words = " ".join(lines).split()
fonts = glob.glob(f"{p.fontdir}/**/*ttf")

def random_paragraph_text():
    start = random.randint(0, len(words)-10000)
    n_rows = random.randint(1,7)
    mode = random.randint(0,4)
    if mode == 0:
        char_width = random.randint(20, 80)
        text = " ".join(words[start: start+1000])
        lines = textwrap.wrap(text, width=char_width)
        return "\n".join(lines[:n_rows])
    elif mode == 1:
        res = []
        for _ in range(n_rows):
            end = start + random.randint(2, 10)
            text = " ".join(words[start:end])
            res.append(text)
            start = end
        return "\n".join(res)
    elif mode == 2:
        return words[start]
    elif mode == 3:
        return "\n".join(words[start : start + n_rows])
    elif mode == 4:
        return " ".join(words[start : start + n_rows])
    else:
        raise ValueError

def random_page_texts():
    return [random_paragraph_text() for _ in range(random.randint(5, 10))]


def draw_multiple_line_text(image, text, font, text_color, text_start_y, text_start_x, align, spacing):
    draw = ImageDraw.Draw(image)
    size = (draw.multiline_textsize(text, font=font))
    draw.multiline_text((text_start_x, text_start_y), text, fill=text_color, font=font, align=align,spacing=spacing)
    return size

def draw_pseudoparagraph(text):
    left_pad, top_pad, right_pad, bottom_pad = random.randint(0, 50), random.randint(0,50), random.randint(0,50), random.randint(0,50),
    tmp_image = Image.new('RGB', (3400, 1000), color=(255, 255, 255))
    ttf = random.choice(fonts)
    fontsize = random.randint(10, 60)
    spacing = random.randint(1, 12)
    font = ImageFont.truetype(ttf, fontsize)
    centering = random.choice(["left", "center", "right"])
    sz = draw_multiple_line_text(tmp_image, text, font, (0, 0, 0), align=centering, text_start_x=left_pad, text_start_y=top_pad, spacing=spacing)
    return tmp_image.crop((0, 0, left_pad+right_pad+sz[0], top_pad+bottom_pad+sz[1]))

def draw_pseudopage(paragraph_texts):
    paragraphs = []
    for paragraph_text in paragraph_texts:
        paragraphs.append(draw_pseudoparagraph(paragraph_text))
    sizes = [img.size for img in paragraphs]
    placements = rpack.pack(sizes)
    new_image_width = max([sizes[n][0]+placements[n][0] for n in range(len(sizes))])
    new_image_height = max([sizes[n][1]+placements[n][1] for n in range(len(sizes))])
    new_image = Image.new('RGB', (new_image_width, new_image_height), color=(255, 255, 255))
    for n in range(len(paragraphs)):
        left,top,right,bottom = placements[n][0],placements[n][1],placements[n][0]+sizes[n][0],placements[n][1]+sizes[n][1]
        new_image.paste(paragraphs[n], (left,top,right,bottom))
    return new_image


if __name__ == "__main__":
    pathlib.Path(p.outdir).mkdir(parents=True, exist_ok=True)
    for n in tqdm.tqdm(range(p.npages)):
        image = draw_pseudopage(random_page_texts())
        image.save(f"{p.outdir}/synth_page{n:05}.png")
