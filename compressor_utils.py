from typing import List

import bitarray as bitarray
import numpy as np
import torch
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel


def save_compressed(compressed: List[int], shape: List[int], B: int, file_name: str):
    with open(file_name, 'wb') as f:
        q = bitarray.bitarray(compressed)
        f.write(f'{B}\n'.encode())
        f.write(f'{"_".join(map(str, shape))}\n'.encode())
        f.write(f'{len(compressed)}\n'.encode())
        f.write(q.tobytes())


def load_compressed(file_name):
    with open(file_name, 'rb') as f:
        compressed = bitarray.bitarray()
        data = f.read()
        B, shape, bytes_cnt, data = data.split(b"\n", 3)
        B = int(B)
        bytes_cnt = int(bytes_cnt)
        shape = list(map(int, shape.split(b"_")))
        compressed.frombytes(data)
        compressed = compressed.tolist()[:bytes_cnt]
    return compressed, shape, B


def get_coder(B: int) -> AECompressor:
    keys = [key for key in range(0, 2 ** B + 1)]
    prob = 1.0 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)
    return coder


def compress(quantized_encoded: torch.Tensor, B: int):
    shape = quantized_encoded.shape
    coder = get_coder(B)
    return coder.compress(quantized_encoded.flatten().tolist()), list(shape)


def decompress(compressed: List[int], shape: List[int], B: int) -> torch.Tensor:
    length = np.prod(shape)
    coder = get_coder(B)
    decompressed = coder.decompress(compressed, length)
    decompressed = np.fromiter(map(int, decompressed), dtype=np.int64)
    decompressed = torch.from_numpy(decompressed).float()
    decompressed = decompressed.reshape(*shape)
    return decompressed
