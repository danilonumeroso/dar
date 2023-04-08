import clrs
import torch

from nn import losses as loss
from nn.models.impl import _dimensions, _expand, _hints_i, _own_hints_i
from nn.models.impl import decoders
from nn.models.impl import encoders
from nn.models.impl import processors
from utils import is_not_done_broadcast
from utils.data import adj_mat, edge_attr_mat
from typing import Callable, Dict, List
from torch.nn import Module, ModuleDict
from torch.nn.functional import relu

Result = Dict[str, clrs.DataPoint]

_Feedback = clrs.Feedback
_Location = clrs.Location
_OutputClass = clrs.OutputClass
_Spec = clrs.Spec
_Stage = clrs.Stage
_Type = clrs.Type
_Type = clrs.Type
_DataPoint = clrs.DataPoint


class EncodeProcessDecode(clrs.Model):
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
        self.net_ = EncodeProcessDecode_Impl(spec=spec,
                                             dummy_trajectory=dummy_trajectory,
                                             num_hidden=num_hidden,
                                             encode_hints=encode_hints,
                                             decode_hints=decode_hints,
                                             processor=processor,
                                             aggregator=aggregator,
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


class EncodeProcessDecode_Impl(Module):
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
                 device: str = 'cpu'):
        super().__init__()

        self.num_hidden = num_hidden
        self.decode_hints = decode_hints

        self.encoders = ModuleDict({})
        self.decoders = ModuleDict({})
        self.hint_decoders = ModuleDict({})

        self.max_steps = max_steps

        self.no_feats = lambda x: x in no_feats or x.startswith('__') # noqa

        for inp in dummy_trajectory.features.inputs:
            if self.no_feats(inp.name):
                continue

            self.encoders[inp.name] = encoders.Encoder(
                in_features=_expand(inp.data, inp.location).shape[-1],
                out_features=self.num_hidden,
                bias=False)

        if encode_hints:
            for hint in dummy_trajectory.features.hints:
                if self.no_feats(hint.name):
                    continue
                self.encoders[hint.name] = encoders.Encoder(
                    in_features=_expand(hint.data[0], hint.location).shape[-1],
                    out_features=self.num_hidden,
                    bias=False)

        self.process = processors.PROCESSORS[processor](num_hidden=num_hidden,
                                                        aggregator=aggregator,
                                                        activation=relu)

        if decode_hints:
            for hint in dummy_trajectory.features.hints:
                if self.no_feats(hint.name):
                    continue
                self.hint_decoders[hint.name] = decoders.new_decoder(spec[hint.name],
                                                                     num_hidden,
                                                                     num_classes=hint.data.shape[-1])

        for out in dummy_trajectory.outputs:
            self.decoders[out.name] = decoders.new_decoder(spec[out.name],
                                                           num_hidden,
                                                           num_classes=out.data.shape[-1])

        self.device = device
        self.spec = spec
        self.encode_hints = encode_hints
        self.to(device)

    def step(self, trajectories, h, adj):
        # ~~~ init ~~~
        batch_size, num_nodes = _dimensions(trajectories[0])
        x = torch.zeros((batch_size, num_nodes, self.num_hidden)).to(self.device)
        edge_attr = torch.zeros((batch_size, num_nodes, num_nodes, self.num_hidden)).to(self.device)

        # ~~~ encode ~~~
        for trajectory in trajectories:
            for dp in trajectory:
                if self.no_feats(dp.name) or dp.name not in self.encoders:
                    continue
                data = encoders.preprocess(dp, num_nodes).to(self.device)
                encoder = self.encoders[dp.name]
                x = encoders.accum_node_fts(encoder, dp, data, x)
                edge_attr = encoders.accum_edge_fts(encoder, dp, data, edge_attr, adj)
                # graph_fts = encoders.accum_graph_fts(encoder, dp, data, graph_fts)

        # ~~~ process ~~~
        z = torch.cat([x, h], dim=-1)
        hiddens = self.process(z, adj, edge_attr)
        h_t = torch.cat([z, hiddens], dim=-1)
        self.h_t = h_t
        self.edge_attr = edge_attr

        # ~~~ decode ~~~
        if not self.decode_hints:
            hint_preds = {}
        else:
            hint_preds = {
                name: decoders.decode_from_latents(
                    name,
                    self.spec[name],
                    self.hint_decoders[name],
                    h_t,
                    adj,
                    edge_attr)
                for name in self.hint_decoders.keys()
            }

        output_preds = {
            name: decoders.decode_from_latents(
                name,
                self.spec[name],
                self.decoders[name],
                h_t,
                adj,
                edge_attr)
            for name in self.decoders.keys()
        }

        return output_preds, hiddens, hint_preds

    def forward(self, features):
        output_preds = {}
        hint_preds = []

        num_steps = self.max_steps if self.max_steps else features.hints[0].data.shape[0] - 1
        batch_size, num_nodes = _dimensions(features.inputs)

        h = torch.zeros((batch_size, num_nodes, self.num_hidden)).to(self.device)

        adj = adj_mat(features).to(self.device)
        A = edge_attr_mat(features).to(self.device)

        for i in range(num_steps):
            cur_hint = _hints_i(features.hints, i) if self.training or i == 0 else _own_hints_i(hint_preds[-1], self.spec, features, i)

            trajectories = [features.inputs]
            if self.encode_hints:
                trajectories.append(cur_hint)

            cand, h, h_preds = self.step(trajectories, h, adj)

            if "f" in cand:
                cand["f"] = A * cand["f"]

            if "f_h" in h_preds:
                h_preds["f_h"] = A * h_preds["f_h"]

            hint_preds.append(h_preds)

            for name in cand:
                if i == 0 or features.lengths.sum() == 0:
                    # if the algorithm has no hints, bypass the following check
                    output_preds[name] = cand[name]
                else:
                    is_not_done = is_not_done_broadcast(features.lengths, i, cand[name])
                    output_preds[name] = is_not_done * cand[name] + \
                        (1.0 - is_not_done) * output_preds[name]

        return output_preds, hint_preds
