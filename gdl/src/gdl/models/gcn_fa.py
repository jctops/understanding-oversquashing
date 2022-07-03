from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset


class GCN_FA(torch.nn.Module):
    def __init__(
        self, dataset: InMemoryDataset, hidden: List[int] = [64], dropout: float = 0.5
    ):
        super(GCN_FA, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-2], num_features[1:-1]):
            layers.append(GCNConv(in_features, out_features))
        
        # Here's the +FA addition
        layers.append(Linear(num_features[-2], num_features[-1]))

        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                x = layer(x, edge_index, edge_weight=edge_attr)
                x = self.act_fn(x)
                x = self.dropout(x)
            else:
                # FA layer!
                x = layer(x)
                x = torch.matmul(torch.ones(x.shape[0], x.shape[0]).cuda(), x)

        return torch.nn.functional.log_softmax(x, dim=1)
