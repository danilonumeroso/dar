import clrs
import torch
from utils.data import adj_mat
from nn.models.impl import _expand_to

_Feedback = clrs.Feedback
_Location = clrs.Location
_OutputClass = clrs.OutputClass
_Spec = clrs.Spec
_Stage = clrs.Stage
_Type = clrs.Type
_DataPoint = clrs.DataPoint

EPS = 1e-12


def cross_entropy(y_pred, y_true, num_classes):
    from torch import mean, sum
    from torch.nn.functional import one_hot, log_softmax
    return mean(-sum(
        one_hot(y_true, num_classes) * log_softmax(y_pred, dim=-1),
        dim=-1
    ), dim=-1)


def mse_loss(y_pred, y_true):
    from torch import mean
    return mean((y_pred - y_true)**2, dim=-1)


def mask_loss(y_pred, y_true):
    from torch import abs, exp, log1p, maximum, zeros_like
    return maximum(y_pred, zeros_like(y_pred)) - y_pred * y_true + log1p(exp(-abs(y_pred)))


def dual_loss(dual_y, inputs, alpha, device='cpu'):
    from torch import mean, sum, take_along_dim
    from torch.linalg import vector_norm

    for inp in inputs:
        if inp.name == 'adj':
            adj = inp.data.to(device)
        elif inp.name == 'A':
            weights = inp.data.to(device)
        elif inp.name == 's':
            source = inp.data.to(device)
        elif inp.name == 't':
            target = inp.data.to(device)

    source_idxs = source.argmax(dim=-1, keepdims=True)
    target_idxs = target.argmax(dim=-1, keepdims=True)

    y_s = take_along_dim(dual_y, source_idxs, dim=1)
    y_t = take_along_dim(dual_y, target_idxs, dim=1)

    dual_y = dual_y.unsqueeze(-1)

    # main objective: max y(t) - y(s)
    e1 = (y_t - y_s).unsqueeze(-1)

    # penalty term constraint: \forall u,v : y(v) - y(u) <= w_uv
    e2 = dual_y.permute((0, 2, 1)) - dual_y
    e2 = e2 * adj
    e2 = (e2 - weights) * (e2 > weights).float()

    # penalty term constraint: \forall u: y(u) >= 0
    e3 = -dual_y * ((dual_y < 0) * 1.0)

    return -e1.mean() + mean(sum(e2, dim=-1)) + mean(sum(e3, dim=-1)) + alpha * vector_norm(e1, ord=2)**2


def max_flow(x, inputs, device='cpu'):

    from torch import bmm, mean, sum

    batch_len, num_nodes, _ = x.shape

    for inp in inputs:
        if inp.name == 's':
            source = inp.data.to(device)
        elif inp.name == 't':
            target = inp.data.to(device)

    # main objective: max sum_{(s,v) \in E} x_sv

    e1 = bmm(source.unsqueeze(1), x).squeeze()
    e2 = bmm(target.unsqueeze(1), x).squeeze()

    e1 = -mean(sum(e1, dim=-1))
    e2 = mean(sum(e2, dim=-1))

    return e1 + e2


def min_cut(S, inputs, device='cpu', reducer=torch.mean):
    import torch

    for inp in inputs:
        if inp.name == 'A':
            capacity = inp.data.to(device)
        elif inp.name == 's':
            source = inp.data.to(device)
        elif inp.name == 't':
            target = inp.data.to(device)

    num_cuts = S.shape[-1]

    S = S.softmax(-1)
    S_t = S.transpose(1, 2)

    l_cut = _3d_trace(S_t @ capacity @ S) / _3d_trace(S_t @ _3d_diag(capacity.sum(-1)) @ S)
    l_ort = torch.linalg.matrix_norm(S_t @ S / torch.linalg.matrix_norm(S_t @ S, keepdims=True) -
                                     torch.eye(num_cuts, device=device) / torch.tensor(num_cuts, device=device).sqrt()
                                     )

    source = torch.bmm(source.unsqueeze(1), S).squeeze()
    target = torch.bmm(target.unsqueeze(1), S).squeeze()
    l_dot = (source * target).sum(-1)  # dot-product

    loss = -l_cut + l_ort + l_dot
    return reducer(loss) if reducer else loss, {
        "l_cut": reducer(-l_cut).detach() if reducer else (-l_cut).detach(),
        "l_ort": reducer(l_ort).detach() if reducer else l_ort.detach(),
        "l_dot": reducer(l_dot).detach() if reducer else l_dot.detach()
    }


def hint_loss(preds, truth, feedback, alpha, device):
    import torch
    import numpy as np

    losses = []
    hint_mask = []
    adj = adj_mat(feedback.features).to(device)

    for i in range(truth.data.shape[0] - 1):
        y = truth.data[i + 1].to(device)

        y_pred = preds[i][truth.name]
        h_mask = (y_pred != clrs.OutputClass.MASKED) * 1.0

        if truth.type_ == _Type.SCALAR:

            loss = (y_pred - (y * adj))**2
            # if truth.name == "f_h":
            #    loss = alpha * loss + (1-alpha) * max_flow(y_pred, feedback.features.inputs, device=device
            if truth.name == "f_h":
                hint_mask.append(h_mask.all(-1).all(-1))
                loss = (loss * h_mask).sum(-1).sum(-1) / adj.sum(-1).sum(-1).to(device)
            else:
                hint_mask.append(h_mask.all(-1))

        elif truth.type_ == _Type.MASK:
            hint_mask.append(h_mask.all(-1))
            loss = mask_loss(y_pred, y)
            mask = (truth.data[i + 1] != _OutputClass.MASKED).float().to(device)
            mask *= h_mask
            loss = torch.sum(loss * mask, dim=-1) / (torch.sum(mask, dim=-1) + EPS)
        elif truth.type_ == _Type.MASK_ONE:
            loss = -torch.sum(y * torch.nn.functional.log_softmax(y_pred, dim=-1) * h_mask, dim=-1)
        elif truth.type_ == _Type.POINTER:
            from torch.nn.functional import log_softmax, one_hot
            hint_mask.append(h_mask.all(-1).all(-1))
            # cross entropy
            loss = one_hot(y.long(), y_pred.shape[-1]) * log_softmax(y_pred, dim=-1)
            loss = -torch.sum(loss * h_mask, dim=-1).mean(-1)

        elif truth.type_ == _Type.CATEGORICAL:
            from torch.nn.functional import log_softmax, one_hot
            # cross entropy
            hint_mask.append(h_mask.all(-1).all(-1))
            loss = one_hot(y.argmax(-1).long(), y_pred.shape[-1]) * log_softmax(y_pred, dim=-1)
            loss = -torch.sum(loss * h_mask, dim=-1).mean(-1)

        losses.append(loss)

    losses = torch.stack(losses)
    hint_mask = torch.stack(hint_mask) * 1.0
    is_not_done = _is_not_done_broadcast(feedback.features.lengths, np.arange(truth.data.shape[0] - 1)[:, None], losses)
    mask = is_not_done * _expand_to(hint_mask, len(is_not_done.shape))

    return (losses * mask).sum() / mask.sum()


def output_loss(preds, truth, feedback, alpha, device):
    import torch
    y_pred = preds[truth.name]
    y = truth.data.to(device)
    adj = adj_mat(feedback.features).to(device)

    if truth.name == "unsup_lp_flow":
        return max_flow(y_pred, feedback.features.inputs, alpha=alpha, device=device)
    elif truth.type_ == _Type.POINTER:
        y = y.long()
        return torch.mean(cross_entropy(y_pred, y, num_classes=y_pred.shape[-1]))
    elif truth.type_ == _Type.CATEGORICAL:
        return torch.mean(cross_entropy(y_pred, y.argmax(-1).long(), num_classes=y_pred.shape[-1]))
    elif truth.location == _Location.EDGE and truth.type_ == _Type.SCALAR:
        loss = ((y_pred - (y * adj))**2).sum() / adj_mat(feedback.features).sum()
        # if truth.name == "f":
        #    loss = alpha * loss + (1-alpha) * max_flow(y_pred, feedback.features.inputs, device=device)

        return loss

    assert False


def _capacity_constraint(pred, inputs, device):
    for inp in inputs:
        if inp.name == 'A':
            capacity = inp.data.to(device)

    return pred * (pred > capacity) * 1.0


def _3d_trace(A):
    return A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def _3d_diag(A):
    return A.diag_embed(offset=0, dim1=-2, dim2=-1)


def _is_not_done_broadcast(lengths, i, tensor):
    import torch
    is_not_done = torch.as_tensor((lengths > i + 1) * 1.0, dtype=torch.float32).to(tensor.device)
    while len(is_not_done.shape) < len(tensor.shape):
        is_not_done = is_not_done.unsqueeze(-1)
    return is_not_done


def _bfs_op_mask(hints):
    for dp in hints:
        if dp.name == '__is_bfs_op':
            return dp.data
