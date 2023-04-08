import clrs
import torch

from nn import losses as loss
from nn.models.impl import _dimensions, _bfs_op_mask, _expand_to, \
    _get_fts, _hints_i, _own_hints_i, _reset_hints
from nn.models.impl import decoders
from nn.models.epd import EncodeProcessDecode_Impl as Net

from random import random
from typing import Callable, Dict, List
from utils import is_not_done_broadcast
from utils.data import adj_mat, edge_attr_mat

Result = Dict[str, clrs.DataPoint]

_INFINITY = 1e5

_Feedback = clrs.Feedback
_Spec = clrs.Spec
_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_DataPoint = clrs.DataPoint


class MF_Net(clrs.Model):
    def __init__(self,
                 spec: _Spec,
                 num_hidden: int,
                 optim_fn: Callable,
                 dummy_trajectory: _Feedback,
                 alpha: float,
                 processor: str,
                 aggregator: str,
                 no_feats: List = ['adj'],
                 add_noise: bool = False,
                 decode_hints: bool = True,
                 encode_hints: bool = True,
                 max_steps: int = None):
        super().__init__(spec=spec)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net_ = MFNet_Impl(spec=spec,
                               dummy_trajectory=dummy_trajectory,
                               processor=processor,
                               aggregator=aggregator,
                               num_hidden=num_hidden,
                               encode_hints=encode_hints,
                               decode_hints=decode_hints,
                               max_steps=max_steps,
                               no_feats=no_feats,
                               add_noise=add_noise,
                               device=self.device)

        self.optimizer = optim_fn(self.net_.parameters())
        self.alpha = alpha
        self.no_feats = lambda x: x in no_feats or x.startswith('__')
        self.encode_hints = encode_hints
        self.decode_hints = decode_hints

    def dump_model(self, path):
        torch.save(self.net_.state_dict(), path)

    def restore_model(self, path, device):
        self.net_.load_state_dict(torch.load(path, map_location=device))

    def _train_step(self, feedback: _Feedback):
        self.net_.train()
        self.optimizer.zero_grad()

        preds, hint_preds = self.net_(feedback.features)
        total_loss = 0.0
        n_hints = 0
        if self.decode_hints:
            hint_loss = 0.0
            for truth in feedback.features.hints:
                if self.no_feats(truth.name):
                    continue

                n_hints += 1
                hint_loss += loss.hint_loss(hint_preds, truth, feedback, self.alpha, self.device)

            total_loss += hint_loss / n_hints

        for truth in feedback.outputs:
            total_loss += loss.output_loss(preds, truth, feedback, self.alpha, self.device)

        total_loss.backward()

        self.optimizer.step()

        return total_loss.item()

    def feedback(self, feedback: _Feedback) -> float:
        loss = self._train_step(feedback)
        return loss

    @torch.no_grad()
    def predict(self, features: clrs.Features) -> Result:
        self.net_.eval()
        raw_preds, aux = self.net_(features)
        preds = decoders.postprocess(raw_preds, self._spec)

        return preds, (raw_preds, aux)

    @torch.no_grad()
    def verbose_loss(self, feedback: _Feedback, preds, aux_preds):
        losses = {}
        total_loss = 0
        n_hints = 0
        for truth in feedback.features.hints:
            if self.no_feats(truth.name):
                continue
            n_hints += 1
            losses["aux_" + truth.name] = loss.hint_loss(aux_preds, truth, feedback, self.alpha, self.device).cpu().item()
            total_loss += losses["aux_" + truth.name]

        total_loss /= n_hints

        for truth in feedback.outputs:
            total_loss += loss.output_loss(preds, truth, feedback, self.alpha, self.device)

        return losses, total_loss.item()


class MFNet_Impl(torch.nn.Module):
    def __init__(self,
                 spec: _Spec,
                 dummy_trajectory: _Feedback,
                 num_hidden: int,
                 encode_hints: bool,
                 decode_hints: bool,
                 processor: str,
                 aggregator: str,
                 no_feats: List,
                 add_noise: bool = False,
                 bias: bool = True,
                 max_steps: int = None,
                 load_path: str = None,
                 annealing: bool = True,
                 device: str = 'cpu'):
        super().__init__()

        self.num_hidden = num_hidden
        self.decode_hints = decode_hints

        self.max_steps = max_steps

        self.no_feats = lambda x: x in no_feats or x.startswith('__')  # noqa
        self.bfs_net = Net(spec,
                           dummy_trajectory,
                           num_hidden,
                           encode_hints,
                           decode_hints,
                           processor,
                           aggregator,
                           no_feats,
                           add_noise=add_noise,
                           device=device)

        self.flow_net = Net(spec,
                            dummy_trajectory,
                            num_hidden,
                            encode_hints,
                            decode_hints,
                            processor,
                            aggregator,
                            no_feats,
                            add_noise=add_noise,
                            device=device)

        c = _get_fts(dummy_trajectory.features.hints, name='c_h')
        if c is not None:
            print(c.data.shape[2])
            self.mincut_net = Net(
                spec,
                dummy_trajectory,
                num_hidden,
                encode_hints=False,
                decode_hints=False,
                processor=processor,
                aggregator=aggregator,
                max_steps=c.data.shape[2] + 1,
                no_feats=no_feats,
                device=device
            )
            del self.flow_net.decoders['c']
            del self.flow_net.hint_decoders['c_h']
            del self.bfs_net.hint_decoders['c_h']
            del self.mincut_net.decoders['f']

        self.is_annealing_enabled = annealing
        self.annealing_state = 0
        self.device = device
        self.spec = spec
        self.encode_hints = encode_hints
        self.to(device)

    def op(self, trajectories, h_bfs, adj, is_bfs_op):

        _, h_bfs, h_preds_bfs = self.bfs_net.step(trajectories, h_bfs, adj)

        cand, _, h_preds_fnet = self.flow_net.step(trajectories, h_bfs.detach(), adj)

        for ignored_key in ['f_h', 'c_h']:
            if ignored_key in self.bfs_net.hint_decoders.keys():
                h_preds_bfs[ignored_key] = h_preds_bfs[ignored_key].new_full(h_preds_bfs[ignored_key].shape, clrs.OutputClass.MASKED)

        for ignored_key in ['reach_h', 'pi_h']:
            if ignored_key in self.flow_net.hint_decoders.keys():
                h_preds_fnet[ignored_key] = h_preds_fnet[ignored_key].new_full(h_preds_fnet[ignored_key].shape, clrs.OutputClass.MASKED)

        if self.decode_hints:
            hint_preds = {}
            idx_bfs = is_bfs_op.flatten().nonzero()
            idx_f = (1 - is_bfs_op).flatten().nonzero()
            for name in h_preds_bfs.keys():
                n_dims = len(h_preds_bfs[name].shape)
                hint_preds[name] = torch.empty_like(h_preds_bfs[name]).to(self.device)
                hint_preds[name].fill_(clrs.OutputClass.MASKED)

                hint_preds[name][idx_bfs] = h_preds_bfs[name][idx_bfs]
                hint_preds[name][idx_f] = h_preds_fnet[name][idx_f]

        # attempt to reset h_bfs
        reset = torch.zeros_like(h_bfs)
        h_bfs = h_bfs.masked_scatter(_expand_to(is_bfs_op.bool(), n_dims), reset)

        assert h_bfs[is_bfs_op.flatten().nonzero()].sum().item() == 0

        for name in cand.keys():
            n_dims = len(cand[name].shape)
            mask = torch.zeros_like(cand[name])
            mask.fill_(clrs.OutputClass.MASKED)
            cand[name] = cand[name].masked_scatter(_expand_to(is_bfs_op.bool(), n_dims), mask)

        return cand, h_bfs, hint_preds

    def forward(self, features):
        output_preds = {}
        hint_preds = []

        num_steps = self.max_steps if self.max_steps else features.hints[0].data.shape[0] - 1
        batch_size, num_nodes = _dimensions(features.inputs)

        h_bfs = torch.zeros((batch_size, num_nodes, self.num_hidden)).to(self.device)

        adj = adj_mat(features).to(self.device)
        A = edge_attr_mat(features).to(self.device)

        def next_hint(i):
            use_teacher_forcing = self.training
            first_step = i == 0

            if self.is_annealing_enabled:
                self.annealing_state += 1
                use_teacher_forcing = use_teacher_forcing and not(random() > (0.999 ** self.annealing_state))

            if use_teacher_forcing or first_step:
                return _hints_i(features.hints, i)
            else:
                return _own_hints_i(last_valid_hints, self.spec, features, i)

        prev_is_flow_op = None
        # if not self.training:
        last_valid_hints = {hint.name: hint.data.to(self.device) for hint in next_hint(0)}
        last_valid_hints['pi_h'] = torch.nn.functional.one_hot(last_valid_hints['pi_h'].long(),
                                                               num_nodes).float()
        del last_valid_hints['__is_bfs_op']

        if self.mincut_net is not None:
            mc_out, _ = self.mincut_net(features)
            output_preds['c'] = mc_out['c']
            last_valid_hints['c_h'] = mc_out['c']

        for i in range(num_steps):
            # ~~~ init ~~~
            trajectories = [features.inputs]
            if self.encode_hints:
                cur_hint = next_hint(i)
                if i > 0 and not self.training:
                    assert prev_is_flow_op is not None
                    first_bfs_step = prev_is_flow_op
                    reach_h, pi_h = _reset_hints(cur_hint, _get_fts(features.inputs, "s").data)
                    for hint in cur_hint:
                        if hint.name == 'reach_h':
                            hint.data[first_bfs_step.flatten().nonzero()] = reach_h.data.to(self.device)[first_bfs_step.flatten().nonzero()]
                        elif hint.name == 'pi_h':
                            hint.data[first_bfs_step.flatten().nonzero()] = pi_h.data.to(self.device)[first_bfs_step.flatten().nonzero()]

                trajectories.append(cur_hint)

            if self.decode_hints:
                is_bfs_op = _bfs_op_mask(next_hint(i)).data.to(self.device)
                is_flow_op = (1 - is_bfs_op)
            else:
                is_bfs_op = None

            cand, h_bfs, h_preds = self.op(trajectories, h_bfs, adj, is_bfs_op)

            if self.mincut_net is not None:
                h_preds['c_h'] = mc_out['c']
            if "f" in cand:
                idx = is_flow_op.flatten().nonzero()
                cand["f"][idx] = (A * cand['f'])[idx]

            if "f_h" in h_preds:
                idx = is_flow_op.flatten().nonzero()
                h_preds["f_h"][idx] = (A * h_preds['f_h'])[idx]

            # if not self.training:
            for name in last_valid_hints.keys():
                is_masked = (h_preds[name] == clrs.OutputClass.MASKED) * 1.0

                last_valid_hints[name] = is_masked * last_valid_hints[name] + (1.0 - is_masked) * h_preds[name]

            hint_preds.append(h_preds)

            for name in cand:
                if name not in output_preds or features.lengths.sum() == 0:
                    output_preds[name] = cand[name]
                else:
                    is_not_done = is_not_done_broadcast(features.lengths, i, cand[name])

                    mask = is_not_done * _expand_to(is_flow_op, len(is_not_done.shape))
                    output_preds[name] = mask * cand[name] + \
                        (1.0 - mask) * output_preds[name]

            prev_is_flow_op = is_flow_op

        return output_preds, hint_preds
