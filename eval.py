import argparse
import os
import sys
import struct
import numpy as np
import pandas as pd
import random
import math
import scipy.stats
import itertools
from itertools import accumulate

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from torchvision import transforms

from models.ftic import FrequencyAwareTransFormer
from models.progressive.net import BLI
from models.entropy_coder import EntropyCoder

from utils_trit_plane import get_Nary_tensor
from utils import cal_psnr, pad, crop, cal_msssim, mkfulldir

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing test images")

    parser.add_argument("--model", type=str, choices=["psnr", "ssim"], default="psnr", help="Evaluation model.")

    parser.add_argument("--ckpt-ic-psnr", type=str, default="checkpoint/FTIC_psnr.pth", help="Path to FTIC backbone PSNR checkpoint")
    parser.add_argument("--ckpt-ic-ssim", type=str, default="checkpoint/FTIC_ssim.pth", help="Path to FTIC backbone MS-SSIM checkpoint")

    parser.add_argument("--ckpt-bli-psnr", type=str, default="checkpoint/BLI_psnr.pth", help="Path to BLI PSNR checkpoint")
    parser.add_argument("--ckpt-bli-ssim", type=str, default="checkpoint/BLI_ssim.pth", help="Path to BLI MS-SSIM checkpoint")

    parser.add_argument("--result-dir", type=str, default="result", help="Path to save results")

    parser.add_argument("--L-list", type=int, nargs=5, default=(4, 4, 6, 16, 24), help="Refinement steps for each trit-plane")

    args = parser.parse_args(argv)
    return args


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        self.image_names = sorted([
            f for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
            and f.lower().endswith(valid_exts)
        ])
        self.image_paths = [os.path.join(data_dir, f) for f in self.image_names]
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img


def return_dpict(args, device):
    net = FrequencyAwareTransFormer()
    net = net.to(device)
    net.eval()
    if args.model == "psnr":
        ckpt = torch.load(args.ckpt_ic_psnr, map_location=device)
    else:
        ckpt = torch.load(args.ckpt_ic_ssim, map_location=device)
    net.load_state_dict(ckpt, strict=False)
    net.update()
    for param in net.parameters():
        param.requires_grad = False
    return net


def encode_dataset(args, device, L):
    mode = 3

    net = BLI(L=L if isinstance(L, int) else 4)
    if args.model == "psnr":
        net.load_state_dict(torch.load(args.ckpt_bli_psnr))
    else:
        net.load_state_dict(torch.load(args.ckpt_bli_ssim))
    net = net.to(device)
    net.eval()

    main_model = return_dpict(args, device)

    rans_coder = EntropyCoder()

    test_dataset = ImageDataset(
        data_dir=args.data_dir,
        transform=transforms.ToTensor(),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )

    metrics_bpp = []
    metrics_psnr = []
    metrics_ssim = []

    for i, batch in enumerate(test_dataloader):
        bpp = []
        l_y_hat = []
        l_psnr = []
        l_ssim = []

        print(f"img{i:02d} ", end="")

        x = batch.to(device)
        x_padded = pad(x, p=128)

        num_pixel = x.shape[0] * x.shape[-2] * x.shape[-1]
        height, width = x.shape[-2], x.shape[-1]

        y, _, z_strings, means_init, scales_init = main_model(x_padded, mode="enc", compress=True)
        _, maxL, l_per_ele, trits = get_Nary_tensor(y, means_init, scales_init)

        bpp_est_total = 8 * (len(z_strings[0]) + 12) / num_pixel
        bpp_pred_total = 8 * len(z_strings[0]) / num_pixel

        str_coded = [z_strings[0]]
        bpp.append(8 * (len(z_strings[0]) + 12) / num_pixel)

        scales_cur, means_cur = scales_init.clone(), means_init.clone()
        ctr_crs = means_init.clone()

        upper_crs = ctr_crs + (mode ** torch.min(l_per_ele, torch.ones_like(l_per_ele) * maxL)) * 0.5
        lower_crs = ctr_crs - (mode ** torch.min(l_per_ele, torch.ones_like(l_per_ele) * maxL)) * 0.5

        for l in range(maxL, 0, -1):
            trits_l = trits[..., maxL - l]
            symbols = trits_l[(l_per_ele >= l).view(-1)]

            if l > 5:
                pmf = net.ternary_pmf(
                    upper_crs[l_per_ele >= l],
                    lower_crs[l_per_ele >= l],
                    scales_cur[l_per_ele >= l],
                    means_cur[l_per_ele >= l],
                )

                strings = rans_coder.compress(symbols, pmf)

                upper_fin, lower_fin = upper_crs.clone(), lower_crs.clone()
                upper_fin[l_per_ele >= l] -= (mode ** (l - 1)) * (2 - symbols)
                lower_fin[l_per_ele >= l] += (mode ** (l - 1)) * symbols

                y_hat = net.expectation(upper_fin, lower_fin, scales_cur, means_cur)
                l_y_hat.append(y_hat.clone())

                bpp_pred = -torch.log2(pmf.gather(dim=1, index=symbols.to(torch.int64).unsqueeze(1))).sum()
                bpp_est = 8 * (len(strings) + 4)

                str_coded += [strings]
                bpp.append(bpp_est / num_pixel)

            else:
                upper_fin, lower_fin = upper_crs.clone(), lower_crs.clone()
                upper_fin[l_per_ele >= l] -= (mode ** (l - 1)) * (2 - symbols)
                lower_fin[l_per_ele >= l] += (mode ** (l - 1)) * symbols

                if isinstance(L, list):
                    net.scheduler.L = L[5 - l]

                y_hats, str_coded_l, scales_cur, means_cur, bpp_pred, bpp_est = net.encode(
                    rans_coder, trits_l,
                    scales_init, means_init,
                    scales_cur, means_cur,
                    upper_crs, lower_crs,
                    upper_fin, lower_fin,
                    l_per_ele, l
                )
                str_coded += str_coded_l
                bpp += [(8 * (len(code_) + 4)) / num_pixel for code_ in str_coded_l]
                l_y_hat += y_hats

            print(f"level {l: 2d} ", end="")

            upper_crs = upper_fin.clone()
            lower_crs = lower_fin.clone()

            bpp_pred_total += bpp_pred / num_pixel
            bpp_est_total += bpp_est / num_pixel

        with open(os.path.join(args.result_dir, f"im{i:02d}.bin"), "wb") as f:
            f.write(struct.pack(">i", int(height)))
            f.write(struct.pack(">i", int(width)))

            for c in str_coded:
                f.write(struct.pack(">i", len(c))) # 4 bytes
                f.write(c)

        for y_hat in l_y_hat[::-1]:
            x_hat = main_model(y_hat, mode="dec").clamp(0, 1)
            x_hat = crop(x_hat, x.shape[-2:])

            l_psnr.append(cal_psnr(x, x_hat))
            l_ssim.append(float(cal_msssim(x, x_hat)))

        metrics_psnr.append(l_psnr)
        metrics_ssim.append(l_ssim)
        bpp = list(accumulate(bpp))[1:]
        metrics_bpp.append(bpp[::-1])

        print(f"\n{i:02d} Total bpp pred: {bpp_pred_total:.4f} est: {bpp_est_total:.4f}... overhead: {bpp_est_total - bpp_pred_total:.4f}")

    len_max = max([len(l_bpp) for l_bpp in metrics_psnr])

    metric_bpp = [l_bpp + [0 for _ in range(len_max - len(l_bpp))] for l_bpp in metrics_bpp]
    metric_bpp = np.stack(metric_bpp, axis=-1)
    metric_bpp = np.concatenate([metric_bpp.mean(axis=-1, keepdims=True), metric_bpp], axis=-1)

    metrics_psnr = [l_psnr + [0 for _ in range(len_max - len(l_psnr))] for l_psnr in metrics_psnr]
    metrics_psnr = np.stack(metrics_psnr, axis=-1)
    metrics_psnr = np.concatenate([metrics_psnr.mean(axis=-1, keepdims=True), metrics_psnr], axis=-1)

    metrics_ssim = [l_ssim + [0 for _ in range(len_max - len(l_ssim))] for l_ssim in metrics_ssim]
    metrics_ssim = np.stack(metrics_ssim, axis=-1)
    metrics_ssim = np.concatenate([metrics_ssim.mean(axis=-1, keepdims=True), metrics_ssim], axis=-1)

    return metric_bpp, metrics_psnr, metrics_ssim


def decode_dataset(args, device, L):
    mode = 3
    multiplier = -scipy.stats.norm.ppf(1e-09 / 2)

    metric_bpp = []
    metrics_psnr = []
    metrics_ssim = []

    net = BLI(L=L if isinstance(L, int) else 4)
    if args.model == "psnr":
        net.load_state_dict(torch.load(args.ckpt_bli_psnr))
    else:
        net.load_state_dict(torch.load(args.ckpt_bli_ssim))
    net = net.to(device)
    net.eval()

    main_model = return_dpict(args, device)

    rans_coder = EntropyCoder()

    test_dataset = ImageDataset(
        data_dir=args.data_dir,
        transform=transforms.ToTensor(),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )

    for i, batch in enumerate(test_dataloader):
        x = batch.to(device)
        num_pixel = x.shape[0] * x.shape[2] * x.shape[3]

        l_y_hat = []
        l_psnr = []
        l_ssim = []

        f =  open(os.path.join(args.result_dir, f"im{i:02d}.bin"), "rb")

        height = struct.unpack(">i", f.read(4))[0]
        width = struct.unpack(">i", f.read(4))[0]

        height_z = int(math.ceil(height / 128) * 128) // 64
        width_z = int(math.ceil(width / 128) * 128) // 64

        str_coded = []
        while True:
            len_str = f.read(4)
            if len_str == b'':
                break
            str_coded.append(f.read(struct.unpack(">i", len_str)[0]))

        f.close()

        bitslen_z = (8 + 4 + len(str_coded[0])) * 8 / num_pixel
        bpp = [bitslen_z + sum([(len(c) + 4) * 8 / num_pixel for c in str_coded[1:idx_c + 1]]) for idx_c in range(1, len(str_coded))]
        metric_bpp.append(bpp[::-1])

        z_hat = main_model.entropy_bottleneck.decompress([str_coded.pop(0)], (height_z, width_z))
        scales_init = main_model.h_scale_s(z_hat).clamp_min(0.04)
        means_init = main_model.h_mean_s(z_hat)

        tail = scales_init * multiplier * 2
        l_per_ele = torch.ceil(torch.log(tail) / torch.log(torch.Tensor([mode]).squeeze())).int()
        l_per_ele = torch.clamp(l_per_ele, 1, l_per_ele.max().item())
        maxL = l_per_ele.max().item()
        if torch.sum(l_per_ele == l_per_ele.max()) < 2:
            maxL = int(l_per_ele.max() - 1)
            l_per_ele = torch.clamp(l_per_ele, 1, maxL)

        T = means_init.new_zeros(*means_init.shape, maxL).int()
        for lvl in range(maxL):
            T[l_per_ele >= maxL - lvl, [lvl]] = 1

        scales_cur, means_cur = scales_init.clone(), means_init.clone()
        ctr_crs = means_init.clone()

        upper_crs = ctr_crs + (mode ** torch.min(l_per_ele, torch.ones_like(l_per_ele) * maxL)) * 0.5
        lower_crs = ctr_crs - (mode ** torch.min(l_per_ele, torch.ones_like(l_per_ele) * maxL)) * 0.5

        for l in range(maxL, 0, -1):
            if l > 5:
                pmf = net.ternary_pmf(
                    upper_crs[l_per_ele >= l],
                    lower_crs[l_per_ele >= l],
                    scales_cur[l_per_ele >= l],
                    means_cur[l_per_ele >= l],
                )
                strings = str_coded.pop(0)
                symbols = rans_coder.decompress(strings, pmf).int().to(device)

                upper_fin, lower_fin = upper_crs.clone(), lower_crs.clone()
                upper_fin[l_per_ele >= l] -= (mode ** (l - 1)) * (2 - symbols)
                lower_fin[l_per_ele >= l] += (mode ** (l - 1)) * symbols

                y_hat = net.expectation(upper_fin, lower_fin, scales_cur, means_cur)
                l_y_hat.append(y_hat.clone())

            else:
                if isinstance(L, list):
                    net.scheduler.L = L[5 - l]
                    cur_L = L[5 - l]
                else:
                    cur_L = L

                y_hats, scales_cur, means_cur, upper_fin, lower_fin = net.decode(
                    rans_coder, [str_coded.pop(0) for _ in range(cur_L)],
                    scales_init, means_init,
                    scales_cur, means_cur,
                    upper_crs, lower_crs,
                    l_per_ele, l
                )
                l_y_hat += y_hats

            upper_crs = upper_fin.clone()
            lower_crs = lower_fin.clone()

        print(f"{i:02d}")

        for y_hat in l_y_hat[::-1]:
            x_hat = main_model(y_hat, mode="dec").clamp(0, 1)
            x_hat = crop(x_hat, (height, width))

            l_psnr.append(cal_psnr(x, x_hat))
            l_ssim.append(float(cal_msssim(x, x_hat)))

        metrics_psnr.append(l_psnr)
        metrics_ssim.append(l_ssim)

    len_max = max([len(l_bpp) for l_bpp in metrics_psnr])

    metric_bpp = [l_bpp + [0 for _ in range(len_max - len(l_bpp))] for l_bpp in metric_bpp]
    metric_bpp = np.stack(metric_bpp, axis=-1)
    metric_bpp = np.concatenate([metric_bpp.mean(axis=-1, keepdims=True), metric_bpp], axis=-1)

    metrics_psnr = [l_psnr + [0 for _ in range(len_max - len(l_psnr))] for l_psnr in metrics_psnr]
    metrics_psnr = np.stack(metrics_psnr, axis=-1)
    metrics_psnr = np.concatenate([metrics_psnr.mean(axis=-1, keepdims=True), metrics_psnr], axis=-1)

    metrics_ssim = [l_ssim + [0 for _ in range(len_max - len(l_ssim))] for l_ssim in metrics_ssim]
    metrics_ssim = np.stack(metrics_ssim, axis=-1)
    metrics_ssim = np.concatenate([metrics_ssim.mean(axis=-1, keepdims=True), metrics_ssim], axis=-1)

    return metric_bpp, metrics_psnr, metrics_ssim


if __name__ == "__main__":
    device = f"cuda:0"
    args = parse_args(sys.argv[1:])

    L_list = list(args.L_list)
    img_list = ImageDataset(args.data_dir).image_names
    out_dir = mkfulldir(args.result_dir)

    with torch.no_grad():
        metrics_bpp, metrics_psnr, metrics_ssim = encode_dataset(args, device, L=L_list)

        metrics = np.stack([metrics_bpp, metrics_psnr, metrics_ssim], axis=-1).reshape(metrics_bpp.shape[0], -1)
        df = pd.DataFrame(metrics)
        nested_idx = [['total bpp', 'total psnr', 'total ms-ssim']] + [['{} bpp'.format(img_list[i]), '{} psnr'.format(img_list[i]), '{} ms-ssim'.format(img_list[i])] for i in range(len(img_list))]
        idx = list(itertools.chain.from_iterable(nested_idx))
        df = pd.concat([pd.DataFrame([idx]), df], ignore_index=True)
        df.to_csv(os.path.join(out_dir, f"metric_enc.csv"), index=False, header=False)

        metrics_bpp, metrics_psnr, metrics_ssim = decode_dataset(args, device, L=L_list)

        metrics = np.stack([metrics_bpp, metrics_psnr, metrics_ssim], axis=-1).reshape(metrics_bpp.shape[0], -1)
        df = pd.DataFrame(metrics)
        nested_idx = [['total bpp', 'total psnr', 'total ms-ssim']] + [['{} bpp'.format(img_list[i]), '{} psnr'.format(img_list[i]), '{} ms-ssim'.format(img_list[i])] for i in range(len(img_list))]
        idx = list(itertools.chain.from_iterable(nested_idx))
        df = pd.concat([pd.DataFrame([idx]), df], ignore_index=True)
        df.to_csv(os.path.join(out_dir, f"metric_dec.csv"), index=False, header=False)
