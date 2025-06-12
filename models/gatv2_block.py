# Python 3.11
"""Two-layer GATv2 – 최신 attention-based GNN."""
from __future__ import annotations

import torch
from torch_geometric.nn import GATv2Conv
import torch_geometric.utils as pyg_utils


class GATv2Block(torch.nn.Module):
    """GAT-v2 (Brody et al., 2021)."""

    def __init__(
        self,
        in_ch: int = 16,
        hid: int = 32,
        out_ch: int = 8,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, hid // heads, heads=heads)
        self.conv2 = GATv2Conv(hid, out_ch, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def get_dummy_input(num_nodes: int = 100, in_ch: int = 16):
    torch.manual_seed(0)
    x = torch.randn((num_nodes, in_ch))
    edge_index = pyg_utils.erdos_renyi_graph(num_nodes, edge_prob=0.1)
    return x, edge_index

