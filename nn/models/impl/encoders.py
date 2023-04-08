import clrs
import torch
from torch.nn import Module, Sequential, Linear

_DataPoint = clrs.DataPoint
_Spec = clrs.Spec
_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_Tensor = torch.FloatTensor


class Encoder(Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):

        super(Encoder, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.net = Sequential(
            Linear(in_features=in_features, out_features=out_features, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


def preprocess(dp: _DataPoint, nb_nodes: int) -> _Tensor:
    from torch.nn.functional import one_hot

    if dp.type_ == _Type.POINTER:
        return one_hot(dp.data.long(), nb_nodes).float()

    return dp.data


def accum_edge_fts(encoder, dp: _DataPoint, data: _Tensor,
                   edge_fts: _Tensor, adj: _Tensor) -> _Tensor:

    encoding = _encode_inputs(encoder, dp, data)

    if dp.location == _Location.NODE and dp.type_ == _Type.POINTER:
        edge_fts += encoding

    elif dp.location == _Location.EDGE:
        assert dp.type_ != _Type.POINTER
        edge_fts += encoding

    return edge_fts


def accum_node_fts(encoder, dp: _DataPoint, data: _Tensor,
                   node_fts: _Tensor) -> _Tensor:

    encoding = _encode_inputs(encoder, dp, data)

    if dp.location == _Location.NODE and dp.type_ != _Type.POINTER:
        node_fts += encoding

    return node_fts


def accum_graph_fts(encoders, dp: _DataPoint, data: _Tensor,
                    graph_fts: _Tensor) -> _Tensor:
    encoding = _encode_inputs(encoders, dp, data)

    if dp.location == _Location.GRAPH and dp.type_ != _Type.POINTER:
        graph_fts += encoding

    return graph_fts


def _encode_inputs(encoder, dp: _DataPoint, data: _Tensor) -> _Tensor:
    if dp.type_ in [_Type.CATEGORICAL]:
        encoding = encoder(data)
    else:
        encoding = encoder(data.unsqueeze(-1))
    return encoding
