import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math

from models.progressive.swin import SwinTransformerBlock


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class DSResBlock(nn.Module):
    def __init__(self, dim_in, dim_out=None, kernel_size=3, stride=1, bias=True):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in

        if dim_in == dim_out and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(dim_in, dim_out, kernel_size=stride, stride=stride, padding=0)

        self.res = nn.Sequential(
            nn.LeakyReLU(),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size, stride, bias),
            nn.LeakyReLU(),
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size, 1, bias),
        )

    def forward(self, x):
        return self.skip(x) + self.res(x)


class SwinBlock(nn.Module):
    def __init__(self, dim_in_q, dim, num_heads, window_size=8, depth=6):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim_in_q, dim),
            *[SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if idx % 2 == 0 else window_size // 2,
                cross=False
            ) for idx in range(depth)]
        )

    def forward(self, x_q):
        x_q = x_q.permute(0, 2, 3, 1).contiguous()

        x_q = self.layers(x_q)

        return x_q.permute(0, 3, 1, 2).contiguous()


class ModelHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.head_y = nn.Sequential(
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, out_ch),
            Rearrange('b c h w -> b h w c')
        )
        self.mlp_y = nn.ModuleList(
                nn.Sequential(
                nn.LayerNorm(out_ch * 3),
                nn.Linear(out_ch * 3, out_ch * 4),
                nn.GELU(),
                nn.Linear(out_ch * 4, out_ch),
                nn.Tanh(),
                Rearrange('b h w c -> b c h w')
            ) for _ in range(6)
        )

        self.head_scales = nn.Sequential(
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, out_ch),
            Rearrange('b c h w -> b h w c'),
        )
        self.mlp_scales = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(out_ch),
                nn.Linear(out_ch, out_ch * 4),
                nn.GELU(),
                nn.Linear(out_ch * 4, out_ch),
                nn.Tanh(),
                Rearrange('b h w c -> b c h w')
            ) for _ in range(6)
        )

        self.head_means = nn.Sequential(
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, in_ch),
            DSResBlock(in_ch, out_ch),
            Rearrange('b c h w -> b h w c')
        )
        self.mlp_means = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(out_ch * 3),
                nn.Linear(out_ch * 3, out_ch * 4),
                nn.GELU(),
                nn.Linear(out_ch * 4, out_ch),
                nn.Tanh(),
                Rearrange('b h w c -> b c h w')
            ) for _ in range(6)
        )

    def forward(self, x, ctr, half_bs):
        tl = ((torch.log(half_bs * 2) / math.log(3)).int()).permute(0, 3, 1, 2).contiguous()

        y = self.head_y(x)
        y = (tl == 0) * self.mlp_y[0](torch.cat([y, ctr, half_bs], dim=-1)) + \
            (tl == 1) * self.mlp_y[1](torch.cat([y, ctr, half_bs], dim=-1)) + \
            (tl == 2) * self.mlp_y[2](torch.cat([y, ctr, half_bs], dim=-1)) + \
            (tl == 3) * self.mlp_y[3](torch.cat([y, ctr, half_bs], dim=-1)) + \
            (tl == 4) * self.mlp_y[4](torch.cat([y, ctr, half_bs], dim=-1)) + \
            (tl >= 5) * self.mlp_y[5](torch.cat([y, ctr, half_bs], dim=-1))

        scales = self.head_scales(x)
        scales = (tl == 0) * self.mlp_scales[0](scales) + \
            (tl == 1) * self.mlp_scales[1](scales) + \
            (tl == 2) * self.mlp_scales[2](scales) + \
            (tl == 3) * self.mlp_scales[3](scales) + \
            (tl == 4) * self.mlp_scales[4](scales) + \
            (tl >= 5) * self.mlp_scales[5](scales)

        means = self.head_means(x)
        means = (tl == 0) * self.mlp_means[0](torch.cat([means, ctr, half_bs], dim=-1)) + \
            (tl == 1) * self.mlp_means[1](torch.cat([means, ctr, half_bs], dim=-1)) + \
            (tl == 2) * self.mlp_means[2](torch.cat([means, ctr, half_bs], dim=-1)) + \
            (tl == 3) * self.mlp_means[3](torch.cat([means, ctr, half_bs], dim=-1)) + \
            (tl == 4) * self.mlp_means[4](torch.cat([means, ctr, half_bs], dim=-1)) + \
            (tl >= 5) * self.mlp_means[5](torch.cat([means, ctr, half_bs], dim=-1))

        return y, scales, means
