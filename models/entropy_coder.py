import time

import torch
from torch import Tensor

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf


class EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method="ans"):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _pmf_to_cdf_tensor(self, pmf, tail_mass, pmf_length, max_length):
        _cdf = torch.cat([pmf, tail_mass], dim=1).clamp_(min=2e-05)
        _cdf = torch.round((_cdf / _cdf.sum(dim=1, keepdim=True)).cumsum(dim=1) * (2 ** 16))
        _cdf = torch.cat([torch.zeros_like(tail_mass), _cdf], dim=1).int()
        return _cdf

    def compress(self, symbols, pmf, return_latency=False):
        if return_latency:
            dict_latency = {}
            t = time.time()

        symbols = symbols.reshape(-1)

        if return_latency:
            quantized_cdf = self._pmf_to_cdf_tensor(
                pmf, pmf.new_zeros(pmf.shape[0], 1), [3 for _ in range(pmf.shape[0])], 3
            )
            dict_latency["make cdf"] = time.time() - t
            t = time.time()
        else:
            quantized_cdf = self._pmf_to_cdf_tensor(
                pmf, pmf.new_zeros(pmf.shape[0], 1), [3 for _ in range(pmf.shape[0])], 3
            )

        strings = self.encode_with_indexes(
            symbols.int().tolist(),
            torch.arange(quantized_cdf.shape[0]).int().tolist(),
            quantized_cdf.tolist(),
            (quantized_cdf.new_ones(quantized_cdf.shape[0]).int() * 5).tolist(),
            quantized_cdf.new_zeros(quantized_cdf.shape[0]).int().tolist(),
        )
        if return_latency:
            dict_latency["entropy code"] = time.time() - t

        if return_latency:
            return strings, dict_latency
        return strings

    def decompress(self, strings, pmf):
        quantized_cdf = self._pmf_to_cdf_tensor(
            pmf, pmf.new_zeros(pmf.shape[0], 1), [3 for _ in range(pmf.shape[0])], 3)

        return torch.tensor(
            self.decode_with_indexes(
                strings,
                torch.arange(quantized_cdf.shape[0]).int().tolist(),
                quantized_cdf.tolist(),
                (quantized_cdf.new_ones(quantized_cdf.shape[0]).int() * 5).tolist(),
                quantized_cdf.new_zeros(quantized_cdf.shape[0]).int().tolist(),
            )
        )


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf
