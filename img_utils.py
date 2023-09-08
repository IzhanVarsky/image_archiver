import os

import numpy as np
from PIL import Image


def is_image_path(img_path: str):
    return img_path.lower().endswith(('.jpg', '.png', '.jpeg'))


def get_mse(orig_img, compressed_img):
    if isinstance(orig_img, str):
        orig_img = Image.open(orig_img).convert('RGB')
    if isinstance(compressed_img, str):
        compressed_img = Image.open(compressed_img).convert('RGB')
    mse = np.mean((np.asarray(orig_img) / 255 -
                   np.asarray(compressed_img) / 255) ** 2)
    return mse


def get_psnr(orig_img, compressed_img):
    mse = get_mse(orig_img, compressed_img)
    res = -10 * np.log10(mse)
    return res


def get_bpp(img_path, im_size=None, no_channel=True):
    bits_cnt = os.path.getsize(img_path) * 8
    channels = 1
    if im_size is None:
        image = Image.open(img_path)
        im_size = np.prod(image.size)
        if not no_channel:
            channels = len(image.getbands())
    return bits_cnt / (im_size * channels)


def find_similar_bpp(name: str, bpp: float):
    tmp_path = "tmp.jpg"
    best_quality = 1
    best_bpp = float("inf")
    for quality in range(1, 100, 1):
        pil_img = Image.open(name)
        pil_img.save(tmp_path, quality=quality, subsampling=0)
        cur_bpp = get_bpp(tmp_path)
        if abs(cur_bpp - bpp) < abs(best_bpp - bpp):
            best_quality = quality
            best_bpp = cur_bpp
    os.remove(tmp_path)
    return best_quality


def print_bpps():
    # root = 'test_images'
    root = 'out_test_jpegs_s0_q20'
    img_names = os.listdir(root)
    for img_name in img_names:
        print(f"Img name: {img_name}")
        b = get_bpp(f'{root}/{img_name}')
        print(f'BPP: {b}')


def print_psnrs():
    orig_root = 'test_images'
    comp_root = 'out_test_jpegs_s0_q20'
    img_names = os.listdir(orig_root)
    for img_name in img_names:
        print(f"Img name: {img_name}")
        b = get_psnr(f'{orig_root}/{img_name}', f'{comp_root}/{img_name.replace(".png", ".jpeg")}')
        print(f'PSNR: {b}')


if __name__ == '__main__':
    # print_bpps()
    print_psnrs()
