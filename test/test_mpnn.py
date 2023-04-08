import os, sys; sys.path.insert(0, os.getcwd())  # noqa: E401, E702

from nn.layers import MpnnConv

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential, Linear


def set_seed(seed):
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


set_seed(0)

dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
num_nodes, num_node_features = data.x.shape

data.edge_attr = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=num_nodes)
data.edge_index = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)
data.x = data.x.unsqueeze(0)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MpnnConv(in_channels=dataset.num_features,
                              edge_channels=data.num_edge_features,
                              mid_channels=16,
                              out_channels=16,
                              net=Sequential(Linear(16, 16)),
                              mid_activation=F.relu,
                              aggregator='max',
                              activation=F.elu)
        self.conv2 = MpnnConv(in_channels=16,
                              edge_channels=data.num_edge_features,
                              mid_channels=16,
                              out_channels=dataset.num_classes,
                              net=Sequential(Linear(16, 16)),
                              mid_activation=F.relu,
                              aggregator='max',
                              activation=None)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(200):
    train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch+1:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
