from __future__ import annotations
import torch
from torchvision.models import resnet18, ResNet18_Weights as W

def get_model() -> torch.nn.Module:
    return resnet18(weights=W.IMAGENET1K_V1).eval()

def get_dummy_input() -> tuple[int, int, int, int]:
    return (1, 3, 224, 224)

