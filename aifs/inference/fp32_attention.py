# Exact fp32 windowed attention for AIFS-ENS, walked in query tiles to fit in GPU memory.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class BandedFP32Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # 4096 queries per tile (~0.8 GiB of scores); 1024 gave the same peak memory, see README
        self.block = 4096

    def forward(self, q, k, v, batch_size, causal=False, window_size=None,
                dropout_p=0.0, softcap=None, alibi_slopes=None):
        length = q.shape[-2]
        out = torch.empty_like(q)
        idx = torch.arange(length, device=q.device, dtype=torch.int32)
        with sdpa_kernel(backends=backends):
            for s in range(0, length, self.block):
                e = min(s + self.block, length)
                # keys this tile of queries can reach
                lo = max(0, s - window_size)
                hi = min(length, e + window_size)
                mask = (idx[s:e].unsqueeze(1) - idx[lo:hi].unsqueeze(0)).abs_() <= window_size
                out[:, :, s:e] = F.scaled_dot_product_attention(
                    q[:, :, s:e], k[:, :, lo:hi], v[:, :, lo:hi], attn_mask=mask, dropout_p=dropout_p)
        return out


def patch(model):
    # Replace every flash-attention module with the fp32 banded one.
    from anemoi.models.layers.attention import MultiHeadSelfAttention
    for m in model.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m.attention = BandedFP32Attention()
