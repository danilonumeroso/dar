import clrs
import numpy as np
from typing import List, Tuple
from norm.performance import Timer


_DataPoint = clrs.DataPoint
_Trajectory = clrs.Trajectory
_Trajectories = List[_Trajectory]


def _maybe_show_progress(length):
    import os

    if 'SHOW_LOADER_PROGRESS' in os.environ:
        import typer
        return typer.progressbar(length)

    return None


def _update(progress):
    if progress is None:
        return

    progress.update(1)


def batch_hints_helper(traj_hints: _Trajectories) -> Tuple[_Trajectory, List[int]]:
    """Batches a trajectory of hints samples along the time axis per probe.

    Unlike i/o, hints have a variable-length time dimension. Before batching, each
    trajectory is padded to the maximum trajectory length.

    Args:
    traj_hints: A hint trajectory of `DataPoints`s indexed by time then probe

    Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`,
    and a |sample| list containing the length of each trajectory.
    """

    max_steps = 0
    assert traj_hints  # non-empty
    for sample_hint in traj_hints:
        for dp in sample_hint:
            assert dp.data.shape[1] == 1  # batching axis
            if dp.data.shape[0] > max_steps:
                max_steps = dp.data.shape[0]

    n_samples = len(traj_hints)
    batched_traj = traj_hints[0]  # construct batched trajectory in-place
    hint_lengths = np.zeros(len(traj_hints))

    for i in range(len(traj_hints[0])):
        hint_i = traj_hints[0][i]
        assert batched_traj[i].name == hint_i.name
        batched_traj[i] = _DataPoint(
            name=batched_traj[i].name,
            location=batched_traj[i].location,
            type_=batched_traj[i].type_,
            data=np.zeros((max_steps, n_samples) + hint_i.data.shape[2:]))
        batched_traj[i].data[:hint_i.data.shape[0], :1] = hint_i.data
        if i > 0:
            assert hint_lengths[0] == hint_i.data.shape[0]
        else:
            hint_lengths[0] = hint_i.data.shape[0]

    progress = _maybe_show_progress(traj_hints[1:])

    for hint_ind, cur_hint in enumerate(traj_hints[1:], start=1):
        for i in range(len(cur_hint)):
            assert batched_traj[i].name == cur_hint[i].name

            batched_traj[i].data[:cur_hint[i].data.shape[0], hint_ind:hint_ind+1] = cur_hint[i].data

            if i > 0:
                assert hint_lengths[hint_ind] == cur_hint[i].data.shape[0]
            else:
                hint_lengths[hint_ind] = cur_hint[i].data.shape[0]

        _update(progress)
    return batched_traj, hint_lengths
