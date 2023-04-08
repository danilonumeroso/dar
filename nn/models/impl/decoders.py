
import clrs
import torch
from torch.nn import Module, Sequential, Linear
from typing import Dict

_INFINITY = 1e5

_DataPoint = clrs.DataPoint
_Spec = clrs.Spec
_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_Tensor = torch.Tensor


class Decoder(Module):
    def __init__(self,
                 in_features,
                 out_features):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.net = Sequential(
            Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x):
        return self.net(x)


class DecoderPair(Module):
    def __init__(self,
                 in_features,
                 out_features):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.first = Sequential(
            Linear(in_features=in_features, out_features=out_features)
        )

        self.second = Sequential(
            Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x):
        return self.first(x), self.second(x)


class DecoderEdge(Module):
    def __init__(self,
                 in_features,
                 e_features,
                 out_features):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.first = Sequential(
            Linear(in_features=in_features, out_features=out_features)
        )

        self.second = Sequential(
            Linear(in_features=in_features, out_features=out_features)
        )

        self.third = Sequential(
            Linear(in_features=e_features, out_features=out_features)
        )

    def forward(self, x, edge_feats):
        return self.first(x), self.second(x), self.third(edge_feats)


def new_decoder(spec, num_hidden, num_classes=None):
    stage, location, type_ = spec

    if location == _Location.NODE:
        if type_ in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            return Decoder(in_features=num_hidden*3, out_features=1)
        if type_ == _Type.POINTER:
            return DecoderPair(in_features=num_hidden*3, out_features=num_hidden)
        if type_ == _Type.CATEGORICAL:
            assert num_classes is not None
            return Decoder(in_features=num_hidden*3, out_features=num_classes)

    if location == _Location.EDGE:
        if type_ == _Type.SCALAR:
            return DecoderEdge(in_features=num_hidden*3,
                               e_features=num_hidden,
                               out_features=1)
        if type_ == _Type.CATEGORICAL:
            assert num_classes is not None
            return DecoderEdge(in_features=num_hidden*3, e_features=num_hidden,
                               out_features=num_classes)

    if location == _Location.GRAPH:
        if type_ in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            return Decoder(in_features=num_hidden*3, out_features=1)

    raise ValueError("Unrecognized specs during decoder creation.")


def decode_from_latents(name, spec, decoder, h_t, adj, edge_attr):

    stage, location, type_ = spec
    if location == _Location.NODE:
        if type_ in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            return decoder(h_t).squeeze(-1)
        if type_ == _Type.POINTER:
            p_1, p_2 = decoder(h_t)
            p = torch.matmul(p_1, torch.permute(p_2, (0, 2, 1)))
            # p = p.masked_fill(~adj.bool(), -_INFINITY)
            return p
        if type_ == _Type.CATEGORICAL:
            return decoder(h_t)

    if location == _Location.EDGE:
        assert edge_attr is not None
        if type_ == _Type.SCALAR:
            p_1, p_2, p_e = decoder(h_t, edge_attr)
            p = (p_1.unsqueeze(-2) + p_2.unsqueeze(-3) + p_e).squeeze() * adj

            if name in ["f", "f_h"]:
                p = p - p.transpose(1, 2)  # TODO: remove and generalise
                p = torch.tanh(p)
            return p

        if type_ == _Type.CATEGORICAL:
            p_1, p_2, p_e = decoder(h_t, edge_attr)
            return p_1.unsqueeze(-2) + p_2.unsqueeze(-3) + p_e

    if location == _Location.GRAPH:
        if type_ in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            g_h = torch.amax(h_t, dim=-2)
            return decoder(g_h).squeeze(-1)

    raise ValueError("Unrecognized specs during decoding from latents.")


def postprocess(preds: Dict[str, _Tensor], spec: _Spec) -> Dict[str, _DataPoint]:
    result = {}
    for name in preds.keys():
        _, location, type_ = spec[name]
        data = preds[name]

        if type_ == _Type.SCALAR:
            data = data
        elif type_ == _Type.MASK:
            data = ((data > 0.0) * 1.0)
        elif type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
            data = torch.nn.functional.one_hot(data.argmax(-1), data.shape[-1]).float()
        elif type_ == _Type.POINTER:
            data = data.argmax(dim=-1)

        result[name] = _DataPoint(name, location, type_, data)
    return result
