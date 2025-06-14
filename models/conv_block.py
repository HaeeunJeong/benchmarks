# Python 3.11
"""Lightweight Conv→Conv→ReLU vision stub."""
from __future__ import annotations
import torch
import torch.nn as nn

class ConvConvReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ───────────────────────────── API ─────────────────────────────
def get_model() -> nn.Module:
    return ConvConvReLU()

def get_dummy_input() -> tuple[int, int, int, int]:
    """Return shape tuple — will be random-filled elsewhere."""
    return (1, 3, 224, 224)

