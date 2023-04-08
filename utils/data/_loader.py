import clrs
import numpy as np
import torch

from ._helpers import batch_hints_helper
from .algorithms import SPECS
from clrs._src.probing import split_stages
from norm.io import load
from pathlib import Path
from torch import as_tensor, index_select
from typing import List, Optional, Tuple, Union
from utils.types import Algorithm

_DataPoint = clrs.DataPoint
_Spec = clrs.Spec
_Type = clrs.Type
_Trajectory = clrs.Trajectory
_Trajectories = List[_Trajectory]
_Features = clrs.Features
_Feedback = clrs.Feedback


def _batch_io(traj_io):
    from clrs._src.samplers import _batch_io as _batch_io_helper

    batched_traj = _batch_io_helper(traj_io)
    for traj in batched_traj:
        traj.data = as_tensor(traj.data, dtype=torch.float32)
    return batched_traj


def _batch_hints(traj_hints):
    batched_traj, hint_lengths = batch_hints_helper(traj_hints)
    for traj in batched_traj:
        traj.data = as_tensor(traj.data, dtype=torch.float32)
    return batched_traj, hint_lengths


def _subsample_data(trajectory, indices, axis=0):
    sampled_traj = []
    for dp in trajectory:
        sampled_data = index_select(dp.data, dim=axis, index=indices)
        sampled_traj.append(_DataPoint(dp.name, dp.location, dp.type_, sampled_data))

    return sampled_traj


def _load_probes(file_name: Path, spec: _Spec):
    inputs = []
    outputs = []
    hints = []

    file_probes = load(file_name)

    for probes in file_probes:
        inp, outp, hint = split_stages(probes, spec)
        inputs.append(inp)
        outputs.append(outp)
        hints.append(hint)

    return inputs, outputs, hints


class Loader:
    def __init__(self, file_name: Union[Path, List[Path]], spec: _Spec):

        if isinstance(file_name, list):
            inputs, outputs, hints = [], [], []
            for name in file_name:
                inp, out, hin = _load_probes(name, spec)
                inputs.extend(inp)
                outputs.extend(out)
                hints.extend(hin)
        else:
            inputs, outputs, hints = _load_probes(file_name, spec)

        # Batch and pad trajectories to max(T).
        self._inputs = _batch_io(inputs)
        self._outputs = _batch_io(outputs)
        self._hints, self._lengths = _batch_hints(hints)
        self._num_samples = len(inputs)
        self._file_name = file_name
        self._spec = spec

    def __add__(self, other):
        return Loader([self._file_name, other._file_name], self._spec)

    def next(self, batch_size: Optional[int] = None) -> _Feedback:
        """Subsamples trajectories from the pre-generated dataset.

        Args:
          batch_size: Optional batch size. If `None`, returns entire dataset.

        Returns:
          Subsampled trajectories.
        """

        if batch_size:
            if batch_size > self._num_samples:
                raise ValueError(
                    f'Batch size {batch_size} > dataset size {self._num_samples}.')

            # Returns a fixed-size random batch.
            raw_indices = np.random.choice(self._num_samples, (batch_size,), replace=True)
            indices = as_tensor(raw_indices, dtype=torch.long)

            inputs = _subsample_data(self._inputs, indices, axis=0)
            outputs = _subsample_data(self._outputs, indices, axis=0)
            hints = _subsample_data(self._hints, indices, axis=1)
            lengths = self._lengths[raw_indices]

        else:
            # Returns the full dataset.
            inputs = self._inputs
            hints = self._hints
            lengths = self._lengths
            outputs = self._outputs

        return _Feedback(_Features(inputs, hints, lengths), outputs)

    def get(self, index: int) -> _Feedback:
        index = torch.LongTensor([index])
        inputs = _subsample_data(self._inputs, index, axis=0)
        outputs = _subsample_data(self._outputs, index, axis=0)
        hints = _subsample_data(self._hints, index, axis=1)
        lengths = self._lengths[index]

        for hint in hints:
            hint.data = hint.data[:int(lengths)]

        return _Feedback(_Features(inputs, hints, lengths), outputs)


def load_dataset(split: str,
                 algorithm: Algorithm,
                 folder: Path) -> Tuple[Loader, _Spec]:

    if algorithm not in SPECS:
        raise NotImplementedError(f"No implementation of algorithm {algorithm}")

    spec = SPECS[algorithm]
    loader = Loader(file_name=folder / f'{split}_{algorithm}.pkl', spec=spec)

    return loader, spec
