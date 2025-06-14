from __future__ import annotations
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights as W

def get_model() -> torch.nn.Module:
    return vit_b_16(weights=W.IMAGENET1K_V1).eval()

def get_dummy_input() -> tuple[int, int, int, int]:
    return (1, 3, 224, 224)

