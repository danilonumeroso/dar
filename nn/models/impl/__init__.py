import clrs
import torch

from . import decoders


_Feedback = clrs.Feedback
_Spec = clrs.Spec
_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_DataPoint = clrs.DataPoint


def _dimensions(inputs):
    for input_ in inputs:
        if input_.location in [clrs.Location.NODE, clrs.Location.EDGE]:
            return input_.data.shape[0], input_.data.shape[1]

    assert False


def _bfs_op_mask(hints):
    for dp in hints:
        if dp.name == '__is_bfs_op':
            return dp


def _hints_i(hints, i):
    hints_i = [_DataPoint(dp.name, dp.location, dp.type_, dp.data[i]) for dp in hints]

    for h in hints_i:
        if h.name == 'f_h':
            zero_c_h = 1 - (h.data == 0).all(-1).all(-1) * 1.0
            break

    for h in hints_i:
        if h.name == 'c_h':
            h.data = h.data * (1 - zero_c_h.unsqueeze(-1).unsqueeze(-1))
            break

    return hints_i


def _own_hints_i(preds, spec, features, i):
    hints = list(decoders.postprocess(preds, spec).values())
    hints.append(_bfs_op_mask(_hints_i(features.hints, i)))
    return hints


def _expand(tensor, loc):
    if loc == _Location.NODE:
        n_dims = 3
    elif loc == _Location.EDGE:
        n_dims = 4
    elif loc == _Location.GRAPH:
        n_dims = 2
    else:
        assert False

    return _expand_to(tensor, n_dims)


def _expand_to(tensor, num_dims):
    while len(tensor.shape) < num_dims:
        tensor = tensor.unsqueeze(-1)

    return tensor


def _get_fts(trajectory, name):
    for dp in trajectory:
        if dp.name == name:
            return dp

    return None


def _reset_hints(hints, source):
    b, n = _dimensions(hints)
    reach_h = torch.zeros((b, n))
    for i, s in enumerate(source):
        reach_h[i][s.argmax().item()] = 1

    pi = torch.stack([torch.arange(n)] * b)
    return [_DataPoint('reach_h', _Location.NODE, _Type.MASK, reach_h),
            _DataPoint('pi_h', _Location.NODE, _Type.POINTER, pi)]
