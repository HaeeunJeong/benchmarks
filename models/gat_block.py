from __future__ import annotations

import torch
from torch_geometric.nn.models import GAT
import torch_geometric.utils as pyg_utils

class GATBlock(torch.nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32, out_channels=8,
                 num_layers=2, heads=2):
        super().__init__()
        self.model = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads
        )

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

def get_model() -> torch.nn.Module:
    return GATBlock()

def get_dummy_input(num_nodes=100, in_channels=16):
    x = torch.randn((num_nodes, in_channels))
    edge_index = pyg_utils.erdos_renyi_graph(num_nodes, edge_prob=0.1)
    return x, edge_index

