import clrs
import numpy as np
import chex
import networkx as nx

from ._specs import SPECS
from clrs._src import probing
from clrs._src.probing import ProbesDict
from typing import Tuple


_Stage = clrs.Stage
_Location = clrs.Location
_Type = clrs.Type
_Array = np.ndarray
_Out = Tuple[_Array, ProbesDict]
_OutputClass = clrs.OutputClass


def a_star(A: _Array, h: _Array, s: int, t: int, max_iter=5000) -> _Out:
    """A* search algorithm (Hart et al., 1968)"""

    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS['a_star'])

    A_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A))
        })

    d = np.zeros(A.shape[0])
    mark = np.zeros(A.shape[0])
    in_queue = np.zeros(A.shape[0])
    f = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    d[s] = 0
    f[s] = h[s]
    in_queue[s] = 1

    probing.push(
        probes,
        _Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'd': np.copy(d),
            'f': np.copy(f),
            'mark': np.copy(mark),
            'in_queue': np.copy(in_queue),
            'u': probing.mask_one(s, A.shape[0])
        })

    while in_queue.any():

        u = np.argsort(f + (1.0 - in_queue) * 1e9)[0]  # drop-in for extract-min

        if in_queue[u] == 0 or u == t:
            break

        in_queue[u] = 0
        mark[u] = 1

        for v in range(A.shape[0]):
            if A[u, v] != 0:
                if mark[v] == 0 or d[u] + A[u, v] < d[v]:
                    pi[v] = u
                    d[v] = d[u] + A[u, v]
                    f[v] = d[v] - h[v]
                    mark[v] = 1
                    in_queue[v] = 1

        probing.push(
            probes,
            _Stage.HINT,
            next_probe={
                'pi_h': np.copy(pi),
                'd': np.copy(d),
                'f': np.copy(f),
                'mark': np.copy(mark),
                'in_queue': np.copy(in_queue),
                'u': probing.mask_one(u, A.shape[0])
            })

    probing.push(probes, _Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
    probing.finalize(probes)

    return pi, probes


def dijkstra(A: _Array, s: int, t: int, early_stop: bool = False) -> _Out:
    """Dijkstra's single-source shortest path (Dijkstra, 1959)."""

    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS['dijkstra'])
    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A))
        })

    d = np.zeros(A.shape[0])
    mark = np.zeros(A.shape[0])
    in_queue = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    d[s] = 0
    in_queue[s] = 1

    probing.push(
        probes,
        _Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'd': np.copy(d),
            'mark': np.copy(mark),
            'in_queue': np.copy(in_queue),
            'u': probing.mask_one(s, A.shape[0])
        })

    for _ in range(A.shape[0]):
        u = np.argsort(d + (1.0 - in_queue) * 1e9)[0]  # drop-in for extract-min

        if in_queue[u] == 0 or (early_stop and u == t):
            break

        mark[u] = 1
        in_queue[u] = 0
        for v in range(A.shape[0]):
            if A[u, v] != 0:
                if mark[v] == 0 and (in_queue[v] == 0 or d[u] + A[u, v] < d[v]):
                    pi[v] = u
                    d[v] = d[u] + A[u, v]
                    in_queue[v] = 1

        probing.push(
            probes,
            _Stage.HINT,
            next_probe={
                'pi_h': np.copy(pi),
                'd': np.copy(d),
                'mark': np.copy(mark),
                'in_queue': np.copy(in_queue),
                'u': probing.mask_one(u, A.shape[0])
            })

    probing.push(probes, _Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
    probing.finalize(probes)

    return pi, probes


def max_flow_lp(adj: _Array, capacity: _Array, s: int, t: int):
    """Max flow LP formulation."""

    chex.assert_rank(adj, 2)
    probes = probing.initialize(SPECS['max_flow_lp'])
    A_pos = np.arange(adj.shape[0])
    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / adj.shape[0],
            's': probing.mask_one(s, adj.shape[0]),
            't': probing.mask_one(t, adj.shape[0]),
            'A': np.copy(capacity),
            'adj': probing.graph(np.copy(adj))
        })

    probing.push(
        probes,
        _Stage.OUTPUT,
        next_probe={
            'unsup_lp_flow': np.empty(1)
        }
    )

    probing.finalize(probes)

    return None, probes


def min_cut_lp(adj: _Array, capacity: _Array, s: int, t: int):
    """Min-Cut LP formulation."""

    chex.assert_rank(adj, 2)
    probes = probing.initialize(SPECS['min_cut_lp'])
    A_pos = np.arange(adj.shape[0])
    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / adj.shape[0],
            's': probing.mask_one(s, adj.shape[0]),
            't': probing.mask_one(t, adj.shape[0]),
            'A': np.copy(capacity),
            'adj': probing.graph(np.copy(adj))
        })

    probing.push(
        probes,
        _Stage.OUTPUT,
        next_probe={
            's': np.array([[1, 0]]).repeat(adj.shape[0], axis=0),
        }
    )

    probing.finalize(probes)

    return None, probes


def max_flow_min_cut_lp(adj: _Array, capacity: _Array, s: int, t: int):

    chex.assert_rank(adj, 2)
    probes = probing.initialize(SPECS['max_flow_min_cut_lp'])
    A_pos = np.arange(adj.shape[0])
    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / adj.shape[0],
            's': probing.mask_one(s, adj.shape[0]),
            't': probing.mask_one(t, adj.shape[0]),
            'A': np.copy(capacity),
            'adj': probing.graph(np.copy(adj))
        })

    probing.push(
        probes,
        _Stage.OUTPUT,
        next_probe={
            'unsup_lp_flow': np.empty(1),
            's': np.array([[1, 0]]).repeat(adj.shape[0], axis=0),
        }
    )

    probing.finalize(probes)

    return None, probes


def _ff_impl(A: _Array, s: int, t: int, probes, w):
    f = np.zeros((A.shape[0], A.shape[0]))
    df = np.array(0)

    C = _minimum_cut(A, s, t)

    def reverse(pi):
        u, v = pi[t], t
        while u != v:
            yield u, v
            v = u
            u = pi[u]

    d = np.zeros(A.shape[0])
    msk = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    d[s] = 0
    msk[s] = 1

    probing.push(
        probes,
        _Stage.HINT,
        next_probe={
            'mask': np.copy(msk),
            'd': np.copy(d),
            'pi_h': np.copy(pi),
            'f_h': np.copy(f),
            'df': np.copy(df),
            'c_h': np.copy(C),
            '__is_bfs_op': np.copy([1])
        })

    while True:
        for _ in range(A.shape[0]):
            prev_d = np.copy(d)
            prev_msk = np.copy(msk)
            for u in range(A.shape[0]):
                for v in range(A.shape[0]):
                    if prev_msk[u] == 1 and A[u, v] - abs(f[u, v]) > 0:
                        if msk[v] == 0 or prev_d[u] + w[u, v] < d[v]:
                            d[v] = prev_d[u] + w[u, v]
                            pi[v] = u
                        msk[v] = 1

            probing.push(
                probes,
                _Stage.HINT,
                next_probe={
                    'pi_h': np.copy(pi),
                    'd': np.copy(prev_d),
                    'mask': np.copy(msk),
                    'f_h': np.copy(f),
                    'df': np.copy(df),
                    'c_h': np.copy(C),
                    '__is_bfs_op': np.copy([1])
                })

            if np.all(d == prev_d):
                break

        if pi[t] == t:
            break

        df = min([
            A[u, v] - f[u, v]
            for u, v in reverse(pi)
        ])

        for u, v in reverse(pi):
            f[u, v] += df
            f[v, u] -= df

        d = np.zeros(A.shape[0])
        msk = np.zeros(A.shape[0])
        pi = np.arange(A.shape[0])
        d[s] = 0
        msk[s] = 1
        probing.push(
            probes,
            _Stage.HINT,
            next_probe={
                'pi_h': np.copy(pi),
                'd': np.copy(d),
                'mask': np.copy(msk),
                'f_h': np.copy(f),
                'df': np.copy(df),
                'c_h': np.copy(C),
                '__is_bfs_op': np.array([0])
            })

    return f, probes


def ford_fulkerson(A: _Array, s: int, t: int):

    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS['ford_fulkerson'])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'w': np.copy(w),
        })

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(
        probes,
        _Stage.OUTPUT,
        next_probe={
            'f': np.copy(f)
        }
    )
    probing.finalize(probes)

    return f, probes


def ford_fulkerson_mincut(A: _Array, s: int, t: int):
    chex.assert_rank(A, 2)
    probes = probing.initialize(SPECS['ford_fulkerson_mincut'])
    A_pos = np.arange(A.shape[0])

    rng = np.random.default_rng(0)

    w = rng.random(size=A.shape)
    w = np.maximum(w, w.T) * probing.graph(np.copy(A))

    probing.push(
        probes,
        _Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            't': probing.mask_one(t, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'w': np.copy(w)
        })

    f, probes = _ff_impl(A, s, t, probes, w)

    probing.push(
        probes,
        _Stage.OUTPUT,
        next_probe={
            'f': np.copy(f),
            'c': _minimum_cut(A, s, t)
        }
    )

    probing.finalize(probes)

    return f, probes


def _minimum_cut(A, s, t):
    C = np.zeros((A.shape[0], 2))

    graph = nx.from_numpy_matrix(A)
    nx.set_edge_attributes(graph, {(i, j): A[i, j] for i, j in zip(*A.nonzero())},
                           name='capacity')

    _, cuts = nx.minimum_cut(graph, s, t)

    for v in cuts[0]:
        C[v][0] = 1

    for v in cuts[1]:
        C[v][1] = 1

    return C


def _masked_array(a):
    a = np.empty_like(a)
    a.fill(_OutputClass.MASKED)
    return a


# TEST PURPOSE

def ek_bfs(A: _Array, s: int, t: int):
    chex.assert_rank(A, 2)
    A_pos = np.arange(A.shape[0])
    f = np.zeros((A.shape[0], A.shape[0]))
    df = np.array(0)

    def reverse(pi):
        u, v = pi[t], t
        while u != v:
            yield u, v
            v = u
            u = pi[u]

    while True:
        probes = probing.initialize(SPECS['ek_bfs'])
        probing.push(
            probes,
            _Stage.INPUT,
            next_probe={
                'pos': np.copy(A_pos) * 1.0 / A.shape[0],
                's': probing.mask_one(s, A.shape[0]),
                't': probing.mask_one(t, A.shape[0]),
                'A': np.copy(A),
                'adj': probing.graph(np.copy(A)),
                'f_h': np.copy(f)
            })

        pi, probes = _bfs_ek_impl(A, s, f, probes)

        probing.push(
            probes,
            _Stage.OUTPUT,
            next_probe={
                'pi': np.copy(pi)
            }
        )

        if pi[t] == t:
            break

        df = min([
            A[u, v] - f[u, v]
            for u, v in reverse(pi)
        ])

        for u, v in reverse(pi):
            f[u, v] += df
            f[v, u] -= df

        probing.finalize(probes)
        yield pi, probes

    probing.finalize(probes)

    yield pi, probes
    return


def _bfs_ek_impl(A: _Array, s: int, f: int, probes):

    reach = np.zeros(A.shape[0])
    reach[s] = 1
    pi = np.arange(A.shape[0])

    probing.push(
        probes,
        _Stage.HINT,
        next_probe={
            'reach_h': np.copy(reach),
            'pi_h': np.copy(pi),
        })

    while True:
        prev_reach = np.copy(reach)

        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if A[i, j] - f[i, j] > 0 and prev_reach[i] == 1:
                    if pi[j] == j and j != s:
                        pi[j] = i
                    reach[j] = 1

        probing.push(
            probes,
            _Stage.HINT,
            next_probe={
                'reach_h': np.copy(reach),
                'pi_h': np.copy(pi),
            })

        if np.all(reach == prev_reach):
            break

    return pi, probes
