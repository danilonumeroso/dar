import torch
from nn.layers import MpnnConv
from torch.nn import Linear, Module, ReLU, Sequential


class MPNN(Module):
    def __init__(self,
                 num_hidden: int,
                 aggregator: str,
                 activation: callable = None,
                 bias: bool = True):

        super(MPNN, self).__init__()

        self.conv = MpnnConv(
            in_channels=num_hidden * 2,
            edge_channels=num_hidden,
            mid_channels=num_hidden,
            out_channels=num_hidden,
            net=Sequential(
                Linear(in_features=num_hidden, out_features=num_hidden),
                ReLU(),
                Linear(in_features=num_hidden, out_features=num_hidden),
            ),
            aggregator=aggregator,
            mid_activation=activation,
            activation=activation,
            bias=bias
        )

    def forward(self, x, adj, edge_attr):
        adj = torch.ones_like(adj).to(adj.device)
        x = self.conv(x, adj, edge_attr)
        return x


class PGN(Module):
    def __init__(self,
                 num_hidden: int,
                 aggregator: str,
                 activation: callable = None,
                 bias: bool = True):
        super(PGN, self).__init__()

        self.conv = MpnnConv(
            in_channels=num_hidden * 2,
            edge_channels=num_hidden,
            mid_channels=num_hidden,
            out_channels=num_hidden,
            net=Sequential(
                Linear(in_features=num_hidden, out_features=num_hidden),
                ReLU(),
                Linear(in_features=num_hidden, out_features=num_hidden),
            ),
            aggregator=aggregator,
            mid_activation=activation,
            activation=activation,
            bias=bias
        )

    def forward(self, x, adj, edge_attr):
        x = self.conv(x, adj, edge_attr)
        return x


PROCESSORS = {
    'pgn': PGN,
    'mpnn': MPNN
}
