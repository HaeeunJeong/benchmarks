# Python 3.11
"""Lightweight torch.matmul vision-independent stub."""
from __future__ import annotations

import torch
import torch.nn as nn


class MatMul(nn.Module):
    """Minimal module that multiplies input by a learnable weight via torch.matmul."""

    def __init__(self, in_features: int = 128, out_features: int = 64):
        super().__init__()
        # Learnable parameter with shape (in_features, out_features)
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., in_features)
        Returns:
            Tensor of shape (..., out_features) resulting from x @ weight
        """
        return torch.matmul(x, self.weight)


# ───────────────────────────── API ─────────────────────────────
def get_model() -> nn.Module:
    """Returns a ready-to-use MatMul model (128 → 64)."""
    return MatMul()


def get_dummy_input() -> tuple[int, int]:
    """Shape tuple for dummy input (to be random-filled elsewhere)."""
    return (1, 128)

