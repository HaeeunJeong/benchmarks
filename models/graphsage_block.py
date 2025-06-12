# Python 3.11
"""Two-layer GraphSAGE – inductive milestone GNN."""
from __future__ import annotations

import torch
from torch_geometric.nn import SAGEConv
import torch_geometric.utils as pyg_utils


class GraphSAGEBlock(torch.nn.Module):
    """Hamilton et al., 2017."""

    def __init__(self, in_ch: int = 16, hid: int = 32, out_ch: int = 8):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid)
        self.conv2 = SAGEConv(hid, out_ch)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def get_dummy_input(num_nodes: int = 100, in_ch: int = 16):
    torch.manual_seed(0)
    x = torch.randn((num_nodes, in_ch))
    edge_index = pyg_utils.erdos_renyi_graph(num_nodes, edge_prob=0.1)
    return x, edge_index

