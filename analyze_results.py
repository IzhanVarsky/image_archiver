import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from compressor_utils import compress, save_compressed, load_compressed, decompress
from img_utils import get_psnr, get_bpp, find_similar_bpp
from my_ae import MyAEArchiver


@torch.inference_mode()
def encode_image(logger, img_path, B, cur_out_log_dir, model_ckpt_path, model_comp_type):
    logger(f"Encoding {img_path}...")
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyAEArchiver(B=B, compression=model_comp_type)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    pil_img = Image.open(img_path).convert('RGB')
    tensor_img = TF.to_tensor(pil_img)
    out = model.encode(tensor_img[None, :], use_fake_quantize=False)[0]
    compressed, shape = compress(out, B)
    save_compressed(compressed, shape, B, f'{cur_out_log_dir}/{file_name}_B_{B}_comp_{model_comp_type}.bin')


@torch.inference_mode()
def encode_image(logger, img_path, B, cur_out_log_dir, model_ckpt_path, model_comp_type):
    logger(f"Encoding {img_path}...")
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyAEArchiver(B=B, compression=model_comp_type)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    pil_img = Image.open(img_path).convert('RGB')
    tensor_img = TF.to_tensor(pil_img)
    out = model.encode(tensor_img[None, :], use_fake_quantize=False)[0]
    compressed, shape = compress(out, B)
    save_compressed(compressed, shape, B, f'{cur_out_log_dir}/{file_name}_B_{B}_comp_{model_comp_type}.bin')


@torch.inference_mode()
def decode_image(logger, img_path, B, cur_out_log_dir, model_ckpt_path, model_comp_type):
    logger(f"Decoding {img_path}...")
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyAEArchiver(B=B, compression=model_comp_type)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    compressed_bin_path = f'{cur_out_log_dir}/{file_name}_B_{B}_comp_{model_comp_type}.bin'
    compressed, shape, B = load_compressed(compressed_bin_path)
    decompressed = decompress(compressed, shape, B)
    out = model.decode(decompressed[None, :], use_dequantize=True)[0]
    out_pil: Image.Image = TF.to_pil_image(out)
    out_path = f'{cur_out_log_dir}/{file_name}_decoded_B_{B}_comp_{model_comp_type}.png'
    out_pil.save(out_path)
    # out_pil.show()
    # mse = get_mse(img_path, out_pil)
    # print(f"MSE: {mse}")
    psnr = get_psnr(img_path, out_path)
    logger(f"PSNR: {psnr}")
    bpp = get_bpp(img_path)
    # logger(f"BPP orig: {bpp}")
    bpp = get_bpp(compressed_bin_path, im_size=np.prod(Image.open(img_path).size))
    logger(f"BPP compressed: {bpp}")

    quality = find_similar_bpp(img_path, bpp)
    logger(f"Chosen quality of JPG: {quality}")
    jpg_path = f'{cur_out_log_dir}/{file_name}_B_{B}_best_bpp.jpg'
    Image.open(img_path).save(jpg_path, quality=quality, subsampling=0)
    bpp = get_bpp(jpg_path)
    logger(f"BPP of JPG: {bpp}")
    # mse = get_mse(img_path, jpg_path)
    # print(f"MSE: {mse}")
    psnr = get_psnr(img_path, jpg_path)
    logger(f"PSNR of JPG: {psnr}")

    bpp = get_bpp(out_path)
    # logger(f"BPP decompressed: {bpp}")


@torch.inference_mode()
def encode_decode_image(img_path):
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    B = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyAEArchiver(B=B)
    ckpt = 'train_logs/1693669446694805377/ImageAutoEncoder_BNone_TestArch5_all_imgs_sz256_best_metric_model.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    pil_img = Image.open(img_path).convert('RGB')
    # pil_img.show()
    tensor_img = TF.to_tensor(pil_img)
    out = model(tensor_img[None, :], use_fake_quantize=True)[0]
    out_pil: Image.Image = TF.to_pil_image(out)
    # out_pil.show()
    out_pil.save(f'{file_name}_decoded.png')


def test_psnrs(logger, B, log_dir, model_ckpt_path,
               test_dir='test_images', model_comp_type='low'):
    log_dir = f'{log_dir}/B={B}'
    os.makedirs(log_dir, exist_ok=True)
    cur_log_file = f'{log_dir}/analyzing_logs.txt'
    if os.path.exists(cur_log_file):
        os.remove(cur_log_file)

    def overloaded_logger(*args):
        logger(*args)
        with open(cur_log_file, 'a+') as f:
            f.write(" ".join(args) + "\n")

    logger("=" * 50)
    overloaded_logger(f"Testing PSNR on model {model_ckpt_path}\n"
                      f"Params:\n"
                      f"B: {B}\n"
                      f"Model compression type: {model_comp_type}\n"
                      f"Test dir: {test_dir}")
    for fname in os.listdir(test_dir):
        overloaded_logger("-" * 20)
        cur_path = f'{test_dir}/{fname}'
        encode_image(overloaded_logger, cur_path, B, log_dir, model_ckpt_path, model_comp_type)
        decode_image(overloaded_logger, cur_path, B, log_dir, model_ckpt_path, model_comp_type)


def test():
    Bs = [1, 2, 4, 8]
    for comp in ['low']:
        for B_loaded in [None]:
            for B_tested in Bs:
                model_path = f'./models/ImageAutoEncoder_B{B_loaded}_Comp{comp.capitalize()}.pth'
                out_path = f'doc_images/{comp}_comp_model/trained_with_B={B_loaded}'
                test_psnrs(print, B_tested, out_path, model_path, model_comp_type=comp)


def test2():
    cur_out_log_dir = 'test_psnr'
    # ckpt = 'train_logs/1693691251817979407/ImageAutoEncoder_BNone_TestArch5_all_imgs_sz256_best_metric_model.pth'
    ckpt = 'train_logs/1693669446694805377/ImageAutoEncoder_BNone_TestArch5_all_imgs_sz256_best_metric_model.pth'
    for B in [
        1,
        2,
        4,
        8,
    ]:
        test_psnrs(print, B, cur_out_log_dir, ckpt)
    # encode_decode_image(f'test_images/{fname}.png')


if __name__ == '__main__':
    test()
    # test2()
