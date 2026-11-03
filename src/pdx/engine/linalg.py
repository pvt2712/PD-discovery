"""Batched products of small matrices.

PyTorch's CPU batched matmul uses a fast built-in kernel only while rows x columns x inner size
stays below 400; above that it calls BLAS once per matrix, which at ten domains is about 35 times
slower (forward and backward). `bmm` splits a product into row and column blocks that stay under
that size, which gives the same result to rounding.
"""
from __future__ import annotations

import torch

_LIMIT = 400


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r, k, c = a.shape[-2], a.shape[-1], b.shape[-1]
    if r * k * c < _LIMIT:
        return a @ b
    blk = max(1, int((( _LIMIT - 1) // k) ** 0.5))
    rows = []
    for i in range(0, r, blk):
        ai = a[..., i:i + blk, :]
        rows.append(torch.cat([ai @ b[..., :, j:j + blk] for j in range(0, c, blk)], -1))
    return torch.cat(rows, -2)
