import argparse
import os
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from compressor_utils import compress, save_compressed
from my_ae import MyAEArchiver


def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Compress Image")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--models_dir", type=str, default='models')
    parser.add_argument("--B_model", type=Optional[str], default=8, choices=[1, 2, 4, 8, None])
    parser.add_argument("--compression_type", type=str, default='high', choices=['low', 'high'])
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Starting compression with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    comp = args.compression_type
    image_path = args.image_path
    device = args.device
    B = args.B
    output_dir = args.output_dir

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    model_ckpt_path = f'{args.models_dir}/ImageAutoEncoder_B{args.B_model}_Comp{comp.capitalize()}.pth'
    model = MyAEArchiver(B=B, compression=comp)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    pil_img = Image.open(image_path).convert('RGB')
    tensor_img = TF.to_tensor(pil_img)
    out = model.encode(tensor_img[None, :], use_fake_quantize=False)[0]
    compressed, shape = compress(out, B)
    os.makedirs(output_dir, exist_ok=True)
    out_path = f'{output_dir}/{file_name}_B_{B}_comp_{comp}.bin'
    save_compressed(compressed, shape, B, out_path)
    print(f"Successfully compressed and saved to: {out_path}")
