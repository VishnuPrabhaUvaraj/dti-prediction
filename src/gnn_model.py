"""
src/gnn_model.py
Graph Convolutional Network for DTI prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class DTI_GNN(nn.Module):
    """
    3-layer Graph Convolutional Network
    Layer 1: each atom sees direct neighbours
    Layer 2: each atom sees 2 hops away
    Layer 3: each atom sees 3 hops away
    Global pooling: summarise all atoms into one vector
    FC layers: make final prediction
    """
    def __init__(self, node_features=15,
                 hidden=128, dropout=0.3):
        super(DTI_GNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden // 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
