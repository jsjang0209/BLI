import math
import torch
import torch.nn as nn

from models.progressive.coding_schedules.RDPrioritized import RDPrioritizedScheduler
from models.progressive.layers import SwinBlock, ModelHead
from models.progressive.compressai_custom import LowerBound


class BLI(nn.Module):
    def __init__(
            self,
            ch_y=320,
            ch_hid=960,
            depth=10,
            num_heads=16,
            window_size=8,
            device="cuda",
            L=4,
    ):
        super().__init__()

        self.attn = SwinBlock(
            dim_in_q=ch_y * 4, dim=ch_hid, depth=depth, num_heads=num_heads, window_size=window_size
        )
        self.dec = ModelHead(ch_hid, ch_y)

        self.scheduler = RDPrioritizedScheduler(L=L, device=device)

        self.bound = LowerBound(1e-9)
        self.scalebound = LowerBound(1e-2)

    def encode(
            self,
            coder, trits,
            scales_init, means_init,
            scales_cur, means_cur,
            upper_crs, lower_crs,
            upper_fin, lower_fin,
            l_per_ele, tl,
    ):
        str_coded = []
        y_hats = []

        bpp_pred = 0
        bpp_est = 0

        for s in range(self.scheduler.L):
            mask_class, num_cls = self.scheduler(
                scales=scales_cur.clone(),
                means=means_cur.clone(),
                upper=upper_crs,
                lower=lower_crs,
                l_per_ele=l_per_ele,
                tl=tl,
                is_train=self.training,
            )

            m1 = mask_class == 1
            m2 = mask_class == 2

            #################### Entropy Coding ######################

            pmf = self.ternary_pmf(
                upper_crs[(l_per_ele >= tl) * m1],
                lower_crs[(l_per_ele >= tl) * m1],
                self.scalebound(scales_cur[(l_per_ele >= tl) * m1]),
                means_cur[(l_per_ele >= tl) * m1],
            )
            symbols = trits[((l_per_ele >= tl) * m1).view(-1)]

            strings = coder.compress(symbols, pmf)
            str_coded.append(strings)

            bpp_pred += -torch.log2(pmf.gather(dim=1, index=symbols.to(torch.int64).unsqueeze(1))).sum()
            bpp_est += 8 * (len(strings) + 4)

            #################### Bound Update ######################

            lb_s = ~m2 * lower_fin + m2 * lower_crs
            ub_s = ~m2 * upper_fin + m2 * upper_crs
            ctr_s = (lb_s + ub_s) * 0.5
            half_bs = 0.5 * (ub_s - lb_s)

            #################### Prediction ######################

            attn = self.attn(
                torch.cat([scales_init, means_init, ctr_s, half_bs.detach()], dim=1)
            )
            y_i, scales_i, means_i = self.dec(
                attn,
                ctr_s.permute(0, 2, 3, 1).contiguous(),
                half_bs.detach().permute(0, 2, 3, 1).contiguous()
            )

            y_hat = ctr_s + y_i * half_bs.detach()
            scales_cur = torch.exp(torch.log(scales_init) + scales_i)
            means_cur = ctr_s + means_i * half_bs.detach()

            y_hats.append(y_hat)

        return y_hats, str_coded, scales_cur, means_cur, bpp_pred, bpp_est

    def decode(
            self,
            coder, str_coded,
            scales_init, means_init,
            scales_cur, means_cur,
            upper_crs, lower_crs,
            l_per_ele, tl
    ):
        y_hats = []

        upper_fin, lower_fin = upper_crs.clone(), lower_crs.clone()

        for s in range(self.scheduler.L):
            mask_class, num_cls = self.scheduler(
                scales=scales_cur.clone(),
                means=means_cur.clone(),
                upper=upper_crs,
                lower=lower_crs,
                l_per_ele=l_per_ele,
                tl=tl,
                is_train=self.training,
            )

            m1 = mask_class == 1
            m2 = mask_class == 2

            #################### Entropy Coding ######################

            pmf = self.ternary_pmf(
                upper_crs[(l_per_ele >= tl) * m1],
                lower_crs[(l_per_ele >= tl) * m1],
                self.scalebound(scales_cur[(l_per_ele >= tl) * m1]),
                means_cur[(l_per_ele >= tl) * m1],
            )
            strings = str_coded.pop(0)

            symbols = coder.decompress(strings, pmf).int().to(scales_cur.device)

            trits_l = torch.ones_like(scales_cur, dtype=torch.int32)
            trits_l[(l_per_ele >= tl) * m1] = symbols

            upper_fin[(l_per_ele >= tl) * m1] -= (3 ** (tl - 1)) * (2 - symbols)
            lower_fin[(l_per_ele >= tl) * m1] += (3 ** (tl - 1)) * symbols

            #################### Bound Update ######################

            lb_s = ~m2 * lower_fin + m2 * lower_crs
            ub_s = ~m2 * upper_fin + m2 * upper_crs
            ctr_s = (lb_s + ub_s) * 0.5
            half_bs = 0.5 * (ub_s - lb_s)

            attn = self.attn(
                torch.cat([scales_init, means_init, ctr_s, half_bs.detach()], dim=1)
            )
            y_i, scales_i, means_i = self.dec(
                attn,
                ctr_s.permute(0, 2, 3, 1).contiguous(),
                half_bs.detach().permute(0, 2, 3, 1).contiguous()
            )

            y_hat = ctr_s + y_i * half_bs.detach()
            scales_cur = torch.exp(torch.log(scales_init) + scales_i)
            means_cur = ctr_s + means_i * half_bs.detach()

            y_hats.append(y_hat)

        return y_hats, scales_cur, means_cur, upper_fin, lower_fin

    def ternary_pmf(self, upper, lower, scales, means):
        bs = (upper.to(torch.float64) - lower.to(torch.float64)) / 3
        pmf = torch.stack(
            [self.integration(
                lower + bs * (i + 1),
                lower + bs * i,
                scales,
                means)
                for i in range(3)
            ],
            dim=-1
        )
        pmf /= pmf.sum(dim=-1, keepdim=True)
        return pmf.to(torch.float32)

    def integration(self, upper, lower, scales, means):
        upper64 = upper.to(torch.float64)
        lower64 = lower.to(torch.float64)
        scales64 = scales.to(torch.float64)
        means64 = means.to(torch.float64)

        u = (upper64 - means64) / scales64
        l = (lower64 - means64) / scales64
        out = 0.5 * (torch.erfc(-u / math.sqrt(2)) - torch.erfc(-l / math.sqrt(2)))

        return self.bound(out).to(torch.float32)

    def expectation(self, upper, lower, scales, means):
        upper64 = upper.to(torch.float64)
        lower64 = lower.to(torch.float64)
        scales64 = scales.to(torch.float64)
        means64 = means.to(torch.float64)

        u = (upper64 - means64) / scales64
        l = (lower64 - means64) / scales64
        F = self.bound(0.5 * (torch.erfc(-u / math.sqrt(2)) - torch.erfc(-l / math.sqrt(2))))

        out = means64 - (scales64 / math.sqrt(2 * math.pi) * (torch.exp(-0.5 * u ** 2) - torch.exp(-0.5 * l ** 2))) / F
        return out.to(torch.float32)
