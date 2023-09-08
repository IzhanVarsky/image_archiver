import math

import torch
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, no_relu=False, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.Identity() if no_relu else nn.ReLU(True),
        )

    def forward(self, x):
        return self.layers(x)


class BasicEncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel=3,
            stride=2,
            padding=1,
    ):
        super(BasicEncoderBlock, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv2d(in_channels, out_channels, no_relu=True,
                        kernel_size=1, stride=stride, padding=0),
        )
        self.layers = nn.Sequential(
            BasicConv2d(in_channels, out_channels,
                        kernel_size=kernel, stride=stride, padding=padding),
            BasicConv2d(out_channels, out_channels, no_relu=True,
                        kernel_size=kernel, stride=1, padding=kernel // 2),
        )

    def forward(self, x):
        out = self.layers(x) + self.downsample(x)
        out = nn.functional.relu(out)
        return out


class BasicDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel=3,
            stride=2,
            padding=1,
            output_padding=1,
    ):
        super(BasicDecoderBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BasicConv2d(in_channels, out_channels,
                        kernel_size=kernel, stride=1, padding=kernel // 2),
        )
        self.transpose_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel, stride, padding, output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            BasicConv2d(out_channels, out_channels,
                        kernel_size=kernel, stride=1, padding=kernel // 2),
        )

    def forward(self, x):
        out = self.transpose_layers(x) + self.upsample(x)
        return out


class MyEncoder(nn.Module):
    def __init__(self, in_channels=3, emb_channels=512, compression='high'):
        super().__init__()
        config_channels = [32, 64, 128, 256, emb_channels]
        if compression == 'high':
            config_channels.insert(0, 16)
        layers = []
        cur_in_channels = in_channels
        for new_c in config_channels:
            layers.append(BasicEncoderBlock(cur_in_channels, new_c))
            cur_in_channels = new_c
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class LastDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel=3):
        super(LastDecoderBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel, stride=1, padding=kernel // 2),
        )
        self.transpose_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel, stride=2, padding=kernel // 2, output_padding=1),
        )

    def forward(self, x):
        out = self.transpose_layers(x) + self.upsample(x)
        return out


class MyDecoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, compression='high'):
        super().__init__()
        config_channels = [256, 128, 64, 32]
        if compression == 'high':
            config_channels.append(8)
        layers = []
        cur_in_channels = in_channels
        for new_c in config_channels:
            layers.append(BasicDecoderBlock(cur_in_channels, new_c))
            cur_in_channels = new_c
        self.decoder = nn.Sequential(
            *layers,
            LastDecoderBlock(config_channels[-1], out_channels),
        )

    def forward(self, x):
        return self.decoder(x)


class MyAEArchiver(nn.Module):
    def __init__(self, in_channels=3, B=8, compression='high'):
        super().__init__()
        emb_channels = 512
        self.compression = compression
        self.encoder = MyEncoder(in_channels, emb_channels, self.compression)
        self.decoder = MyDecoder(emb_channels, out_channels=in_channels, compression=self.compression)
        self.B = B
        self.mean = -0.5
        self.std = 0.5

    def forward(self, x, use_fake_quantize=True):
        out = self.encode(x, use_fake_quantize=use_fake_quantize)
        out = self.decode(out)
        return out

    def encode(self, x, use_fake_quantize=True):
        out = self.encoder(x)
        out = torch.sigmoid(out)
        if use_fake_quantize:
            out = self.fake_quantize(out)
        else:
            out = self.encode_true_quantize(out)
        return out

    def decode(self, x, use_dequantize=False):
        if use_dequantize:
            x = self.dequantize(x)
        out = self.decoder(x)
        out = torch.sigmoid(out)
        return out

    def fake_quantize(self, encoded):
        if self.B is None or self.B == 0:
            return encoded
        noise = torch.normal(mean=self.mean, std=self.std, size=tuple(encoded.size())) * math.pow(2, -self.B)
        noise = noise.to(encoded.device)
        encoded = encoded + noise
        return encoded

    def encode_true_quantize(self, encoded):
        if self.B is None or self.B == 0:
            return encoded
        return (encoded * math.pow(2, self.B) + 0.5).to(torch.int64)

    def dequantize(self, encoded):
        return encoded.float() / math.pow(2, self.B)
