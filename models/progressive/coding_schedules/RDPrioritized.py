import math
import torch

from models.progressive.coding_schedules.baseline import CodingScheduler


class RDPrioritizedScheduler(CodingScheduler):
    def __init__(self, **kwargs):
        super().__init__()

        self.L = kwargs["L"]
        self.mask_iter = []
        self.mask_step = []

        self.interpolate = interpolate_rdp

    def __call__(self, scales, l_per_ele, tl, **kwargs):
        upper = kwargs["upper"]
        lower = kwargs["lower"]
        means = kwargs["means"]

        mask_cls, num_cls = self._generate_mask(scales, means, upper, lower, l_per_ele, tl)

        self._count()

        return mask_cls, num_cls

    def _generate_mask(self, scales, means, upper, lower, l_per_ele, tl):
        if self.count == 0:
            masks_batch = [self._create_batch_mask_cls(scales[b], means[b], upper[b], lower[b], l_per_ele[b], tl) for b in range(scales.size(0))]
            self.mask_iter = [torch.stack([mb[l] for mb in masks_batch], dim=0) for l in range(self.L)]

            self.mask_step = []
            for l in range(self.L):
                mask_step = torch.zeros_like(self.mask_iter[l])
                mask_step[self.mask_iter[l] == 1] = 1
                self.mask_step.append(mask_step)

        mask_class = self.mask_iter[self.count]
        num_cls = [int((mask_class == i).sum()) for i in range(self.num_types)]

        return mask_class.int(), num_cls

    @torch.no_grad
    def _create_batch_mask_cls(self, scales, means, upper, lower, l_per_ele, tl):
        ele_mask = l_per_ele >= tl
        rd, delta_R, delta_D = RDPriority(means, scales, lower, upper)
        rd_cpu = rd[ele_mask].detach().to(device="cpu", dtype=torch.float64)
        delta_R_cpu = delta_R[ele_mask].detach().to(device="cpu", dtype=torch.float64)
        order_cpu = torch.argsort(rd_cpu, descending=True, stable=True)
        itp = self.interpolate(self.L, order_cpu, delta_R_cpu)

        orders_sort = order_cpu.to(scales.device)

        mask_batch = []
        for l in range(self.L):
            mask_l = torch.zeros_like(scales)
            mask_l_ele = torch.zeros_like(orders_sort, dtype=torch.float32)

            tmp_mask = torch.zeros_like(mask_l_ele, dtype=torch.bool)
            idx_sort = orders_sort[itp[l]:itp[l + 1]]
            tmp_mask[idx_sort] = True
            mask_l_ele[tmp_mask] = 1

            tmp_mask = torch.zeros_like(mask_l_ele, dtype=torch.bool)
            idx_sort = orders_sort[itp[l + 1]:]
            tmp_mask[idx_sort] = True
            mask_l_ele[tmp_mask] = 2

            mask_l[ele_mask] = mask_l_ele

            mask_batch.append(mask_l)

        return mask_batch

    def _count(self):
        self.count += 1
        if self.count == self.L:
            self.count = 0
            self.mask_iter = []


def interpolate_rdp(L, order, delta_R):
    amounts_sorted = delta_R[order].to(torch.float64)
    cumsum = torch.cumsum(amounts_sorted, dim=0)
    total = cumsum[-1].item() if cumsum.numel() > 0 else 0.0

    targets = torch.linspace(0.0, total, L + 1, dtype=torch.float64, device="cpu")[1:-1]
    split_indices = torch.searchsorted(cumsum, targets).tolist()

    return [0] + split_indices + [len(order)]


def gaussian_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def gaussian_integral(mu, sigma, a, b):
    z_a = (a - mu) / (sigma * math.sqrt(2))
    z_b = (b - mu) / (sigma * math.sqrt(2))
    return (0.5 * (torch.erf(z_b) - torch.erf(z_a))).clamp_min(1e-9)


def gaussian_x_integral(mu, sigma, a, b):
    p_a = gaussian_pdf(a, mu, sigma)
    p_b = gaussian_pdf(b, mu, sigma)
    return mu * gaussian_integral(mu, sigma, a, b) - (sigma ** 2) * (p_b - p_a)


def gaussian_x2_integral(mu, sigma, a, b):
    p_a = gaussian_pdf(a, mu, sigma)
    p_b = gaussian_pdf(b, mu, sigma)
    return (mu ** 2 + sigma ** 2) * gaussian_integral(mu, sigma, a, b) - (sigma ** 2) * ((b + mu) * p_b - (a + mu) * p_a)


def RDPriority(means, scales, a, b):
    # divide into 3 equal parts
    a1 = a + (b - a) / 3
    a2 = b - (b - a) / 3
    intervals = [(a, a1), (a1, a2), (a2, b)]

    # whole interval metrics
    P = gaussian_integral(means, scales, a, b)
    Px = gaussian_x_integral(means, scales, a, b)
    Px2 = gaussian_x2_integral(means, scales, a, b)
    m_old = Px / P
    D_old = Px2 / P - m_old ** 2

    # compute for each subinterval
    D_new = 0
    delta_R = 0
    for s, e in intervals:
        p = gaussian_integral(means, scales, s, e)
        px = gaussian_x_integral(means, scales, s, e)
        px2 = gaussian_x2_integral(means, scales, s, e)

        m_new = px / p
        d_new = (px2 - (m_new ** 2) * p) / P
        D_new += d_new

        p_norm = p / P
        delta_R += -p_norm * torch.log2(p_norm)

    delta_D = (D_new - D_old).clamp_max(0)
    delta_R = delta_R * (delta_R >= 0)

    return (((-delta_D) ** 0.7 / delta_R)).clamp_min(0), delta_R, delta_D
