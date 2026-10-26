import torch
import scipy

multiplier = -scipy.stats.norm.ppf(1e-09 / 2)
mode = 3
opt_pnum = 5


__all__ = [
    "opt_pnum",
    "mode",
    "get_Nary_tensor",
]


def get_empty_Nary_tensor(scales_hat):
    device = scales_hat.device
    scales_hat = torch.clamp(scales_hat, min=0.04)
    tail = scales_hat * multiplier * 2
    l_ele = torch.ceil(torch.log(tail) / torch.log(torch.Tensor([mode]).squeeze())).int()
    l_ele = torch.clamp(l_ele, 1, l_ele.max().item())
    maxL = l_ele.max().item()

    if torch.sum(l_ele == l_ele.max()) < 2:
        maxL = l_ele.max().item() - 1
        l_ele = torch.clamp(l_ele, 1, maxL)

    Nary_tensor = torch.zeros(list(scales_hat.shape) + [maxL]).int().to(device)
    return device, maxL, l_ele, Nary_tensor


def get_Nary_tensor(y, means_hat, scales_hat):
    device, maxL, l_ele, Nary_tensor = get_empty_Nary_tensor(scales_hat)

    half_width = (float(mode) ** l_ele.float()) * 0.5
    y = torch.clamp(y, means_hat - half_width, means_hat + half_width)

    symbol_tensor = torch.round(y - means_hat).int() + torch.div(mode ** l_ele, 2, rounding_mode="floor")
    symbol_tensor = torch.clamp(symbol_tensor, min=torch.zeros(y.shape).int().to(device), max=3 ** l_ele - 1)

    for i in range(1, maxL + 1):
        Nary_tensor[:, :, :, :, i - 1] = torch.div(symbol_tensor, (mode ** (maxL - i)), rounding_mode="floor")
        symbol_tensor = symbol_tensor % (mode ** (maxL - i))

    Nary_tensor = Nary_tensor.view(-1, maxL)
    del symbol_tensor
    torch.cuda.empty_cache()

    return device, maxL, l_ele, Nary_tensor
