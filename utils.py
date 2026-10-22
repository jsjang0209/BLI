import os
import math
import torch
import torch.nn.functional as F

from pytorch_msssim import ms_ssim


def mkdir_(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mkfulldir(path):
    if not isinstance(path, str):
        raise Exception('not str')

    if path.endswith(f".png"):
        path_short = mkfulldir(os.path.dirname(path))
        return path

    if not '/' in path:
        mkdir_(path)
    else:
        subpath_list = path.split("/")
        subpath = subpath_list[0]
        if subpath != "":
            mkdir_(subpath)
        del subpath_list[0]

        for p in subpath_list:
            subpath = "/".join([subpath, p])
            mkdir_(subpath)

    return path

def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
    )

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )

def cal_psnr(a, b):
    return (-10 * torch.log10(((a - b) ** 2).mean(dim=[1, 2, 3]))).mean().item()


def cal_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())
