import torch
from typing import Callable, List
from torch.nn import Linear, Module, Sequential
from torch.nn import functional as F

Inf = 1e6


class MpnnConv(Module):
    def __init__(self,
                 in_channels: int,
                 edge_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 net: Sequential,
                 aggregator: str,
                 mid_activation: Callable = None,
                 activation: Callable = None,
                 bias: bool = True):

        super(MpnnConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.m_1 = Linear(in_features=in_channels,
                          out_features=mid_channels,
                          bias=bias)
        self.m_2 = Linear(in_features=in_channels,
                          out_features=mid_channels,
                          bias=bias)
        self.m_e = Linear(in_features=edge_channels,
                          out_features=mid_channels,
                          bias=bias)

        self.o1 = Linear(in_features=in_channels,
                         out_features=out_channels,
                         bias=bias)
        self.o2 = Linear(in_features=mid_channels,
                         out_features=out_channels,
                         bias=bias)

        self.net = net
        self.mid_activation = mid_activation
        self.activation = activation

        self.aggregator = aggregator

        if aggregator == 'max':
            self.reduce = torch.amax
        elif aggregator == 'sum':
            self.reduce = torch.sum
        elif aggregator == 'mean':
            self.reduce = torch.mean
        else:
            raise NotImplementedError("Invalid type of aggregator function.")

        self.reset_parameters()

    def reset_parameters(self):
        self.m_1.reset_parameters()
        self.m_2.reset_parameters()
        self.m_e.reset_parameters()
        self.o1.reset_parameters()
        self.o2.reset_parameters()

    def forward(self, x, adj, edge_attr):
        """
        x : Tensor
            node feature matrix (batch_size x num_nodes x num_nodes_features)
        adj : Tensor
            adjacency matrix (batch_size x num_nodes x num_nodes)
        edge_attr : Tensor
            edge attributes (batch_size x num_nodes x num_nodes x num_edge_features)
        """

        batch_size, num_nodes, num_features = x.shape
        _, _, _, num_edge_features = edge_attr.shape

        msg_1 = self.m_1(x)
        msg_2 = self.m_2(x)
        msg_e = self.m_e(edge_attr)

        msg = (
            msg_1.unsqueeze(1) +
            msg_2.unsqueeze(2) +
            msg_e
        )
        if self.net is not None:
            msg = self.net(F.relu(msg))

        if self.mid_activation is not None:
            msg = self.mid_activation(msg)

        if self.aggregator == "mean":
            msg = (msg * adj.unsqueeze(-1)).sum(1)
            msg = msg / torch.sum(adj, dim=-1, keepdims=True)
        elif self.aggregator == "max":
            max_arg = torch.where(adj.unsqueeze(-1).bool(),
                                  msg,
                                  torch.tensor(-Inf).to(msg.device))
            msg = self.reduce(max_arg, dim=1)
        else:
            msg = self.reduce(msg * adj.unsqueeze(-1), dim=1)

        h_1 = self.o1(x)
        h_2 = self.o2(msg)

        out = h_1 + h_2

        if self.activation is not None:
            out = self.activation(out)

        return out


class SparseMpnnConv(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 edge_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 net: Sequential,
                 aggregator: str,
                 devices: List[str] = ['cpu', 'cpu'],
                 mid_activation: Callable = None,
                 activation: Callable = None,
                 bias: bool = True):

        super().__init__()

        from torch_geometric.nn.aggr import MaxAggregation

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.m_1 = Linear(in_features=in_channels,
                          out_features=mid_channels,
                          bias=bias)
        self.m_2 = Linear(in_features=in_channels,
                          out_features=mid_channels,
                          bias=bias)
        self.m_e = Linear(in_features=edge_channels,
                          out_features=mid_channels,
                          bias=bias)

        self.o1 = Linear(in_features=in_channels,
                         out_features=out_channels,
                         bias=bias)
        self.o2 = Linear(in_features=mid_channels,
                         out_features=out_channels,
                         bias=bias)

        self.net = net
        self.mid_activation = mid_activation
        self.activation = activation
        self.devices = devices
        self.aggregator = aggregator

        assert aggregator == 'max', "Invalid type of aggregator function."
        self.reduce = MaxAggregation()

    def reset_parameters(self):
        self.m_1.reset_parameters()
        self.m_2.reset_parameters()
        self.m_e.reset_parameters()
        self.o1.reset_parameters()
        self.o2.reset_parameters()

    def forward(self, x, adj_t, edge_attr, reporter=None):
        """
        x : Tensor
            node feature matrix (num_nodes x num_nodes_features)
        adj_t : SparseTensor | Tensor
            sparse adjacency matrix (num_nodes x num_nodes)
        edge_attr : Tensor
            edge attributes (num_edges x num_edge_features)
        """

        x = x.to(self.devices[0])
        edge_attr = edge_attr.to(self.devices[0])
        adj_t = adj_t.to(self.devices[1])

        msg_1 = self.m_1.to(self.devices[0])(x)
        msg_2 = self.m_2.to(self.devices[0])(x)
        msg_e = self.m_e.to(self.devices[0])(edge_attr)

        # for each (u, v) : m1(u) + m2(v) + m_e(uv)

        msg = torch.relu(
            msg_1[adj_t.storage.row()] +
            msg_2[adj_t.storage.col()] +
            msg_e
        )

        msg = self.net.to(self.devices[1])(
            msg.to(self.devices[1])
        ).to(self.devices[1])

        msg = self.reduce(msg,
                          adj_t.storage.row(),
                          ptr=adj_t.storage.rowptr(),
                          dim_size=adj_t.storage.row().max().item() + 1,
                          dim=-2)

        h_1 = self.o1.to(self.devices[1])(x.to(self.devices[1]))
        h_2 = self.o2.to(self.devices[1])(msg)

        out = h_1 + h_2

        if self.activation is not None:
            out = self.activation(out)

        return out


# class SparseMpnnConv(MessagePassing):
#     def __init__(self,
#                  in_channels: int,
#                  edge_channels: int,
#                  mid_channels: int,
#                  out_channels: int,
#                  net: Sequential,
#                  aggregator: str,
#                  mid_activation: Callable = None,
#                  activation: Callable = None,
#                  bias: bool = True):

#         super().__init__(aggr=aggregator)

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.m_1 = Linear(in_features=in_channels,
#                           out_features=mid_channels,
#                           bias=bias)
#         self.m_2 = Linear(in_features=in_channels,
#                           out_features=mid_channels,
#                           bias=bias)
#         self.m_e = Linear(in_features=edge_channels,
#                           out_features=mid_channels,
#                           bias=bias)

#         self.o1 = Linear(in_features=in_channels,
#                          out_features=out_channels,
#                          bias=bias)
#         self.o2 = Linear(in_features=mid_channels,
#                          out_features=out_channels,
#                          bias=bias)

#         self.net = net
#         self.mid_activation = mid_activation
#         self.activation = activation

#         self.aggregator = aggregator

#         if aggregator == 'max':
#             self.reduce = torch.amax
#         elif aggregator == 'sum':
#             self.reduce = torch.sum
#         elif aggregator == 'mean':
#             self.reduce = torch.mean
#         else:
#             raise NotImplementedError("Invalid type of aggregator function.")

#     def reset_parameters(self):
#         self.m_1.reset_parameters()
#         self.m_2.reset_parameters()
#         self.m_e.reset_parameters()
#         self.o1.reset_parameters()
#         self.o2.reset_parameters()

#     def message(self, x_i, x_j, e_ij):
#         msg = self.m_1(x_i) + self.m_2(x_j) + self.m_e(e_ij)
#         msg = self.net(F.relu(msg))
#         return msg

#     def forward(self, x, adj_t, edge_attr):
#         """
#         x : Tensor
#             node feature matrix (num_nodes x num_nodes_features)
#         adj_t : SparseTensor | Tensor
#             sparse adjacency matrix (num_nodes x num_nodes) |
#             coo coordinates tensor (2 x num_edges)
#         edge_attr : Tensor
#             edge attributes (num_edges x num_edge_features)
#         """

#         msg = self.propagate(edge_index=adj_t,
#                              x=x,
#                              e_ij=edge_attr)

#         h_1 = self.o1(x)
#         h_2 = self.o2(msg)

#         out = h_1 + h_2

#         if self.activation is not None:
#             out = self.activation(out)

#         return out
