from __future__ import annotations
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights as W

def get_model() -> torch.nn.Module:
    return mobilenet_v3_small(weights=W.IMAGENET1K_V1).eval()

def get_dummy_input() -> tuple[int, int, int, int]:
    return (1, 3, 224, 224)

