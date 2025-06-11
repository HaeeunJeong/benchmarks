# Python 3.11
import torch.nn as nn

class ConvConvReLU(nn.Module):
    """Two 3×3 conv + ReLU block, C_in=C_out=64"""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

