import numpy as np
import networkx as nx
from itertools import product
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional


def _make_weights_undirected(w):
    w = np.triu(w)
    return np.triu(w) + np.triu(w, 1).T


def _erdos_renyi(num_nodes: int,
                 prob: float,
                 directed: bool = False,
                 weighted: bool = False,
                 rng: Optional[Generator] = None) -> NDArray:

    assert num_nodes >= 0 and 0 < prob <= 1

    if rng is None:
        rng = default_rng()

    adj_matrix = rng.random((num_nodes, num_nodes)) <= prob

    if not directed:
        adj_matrix = adj_matrix + adj_matrix.T

    weights = None
    if weighted:
        weights = rng.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes))
        if not directed:
            weights = _make_weights_undirected(weights)

    return adj_matrix, weights


def erdos_renyi_full(num_nodes: int,
                     prob: float,
                     directed: bool = False,
                     weighted: bool = False,
                     rng: Optional[Generator] = None) -> NDArray:

    adj_matrix, weights = _erdos_renyi(num_nodes=num_nodes,
                                       prob=prob,
                                       directed=directed,
                                       weighted=weighted,
                                       rng=rng)

    adj_matrix = adj_matrix.astype(dtype=np.float32)

    return adj_matrix * weights if weighted else adj_matrix


def two_community(num_nodes: int,
                  prob: float,
                  outer_prob: float,
                  directed: bool = False,
                  weighted: bool = False,
                  rng: Optional[Generator] = None) -> NDArray:

    assert num_nodes % 2 == 0
    adj_matrix_1, _ = _erdos_renyi(num_nodes=num_nodes // 2,
                                   prob=prob,
                                   directed=directed,
                                   weighted=weighted,
                                   rng=rng)
    adj_matrix_2, _ = _erdos_renyi(num_nodes=num_nodes // 2,
                                   prob=prob,
                                   directed=directed,
                                   weighted=weighted,
                                   rng=rng)

    adj_matrix = np.zeros((num_nodes, num_nodes))

    N = num_nodes // 2
    adj_matrix[:N, :N] = adj_matrix_1
    adj_matrix[N:, N:] = adj_matrix_2

    cart = list(product(range(N), range(N, num_nodes)))
    mask = rng.binomial(n=1, p=outer_prob, size=len(cart))

    n = 0
    while mask.sum() == 0:  # prevent disconnected graph
        mask = rng.binomial(n=1, p=outer_prob + (0.01 * n), size=len(cart))
        n += 1

    for i, e in enumerate(cart):
        if mask[i]:
            u, v = e
            adj_matrix[u, v] = 1.

    if not directed:
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    return adj_matrix


def bipartite(num_nodes: int,
              prob: float,
              outer_prob: float = None,  # unused param
              directed: bool = True,
              weighted: bool = False,
              rng: Optional[Generator] = None) -> NDArray:

    if rng is None:
        rng = default_rng()

    N = (num_nodes-2) // 2

    adj_matrix = np.zeros((num_nodes, num_nodes))

    set_a = list(range(1, N+1))
    set_b = list(range(N+1, num_nodes-1))

    cart = list(product(set_a, set_b))
    mask = rng.binomial(n=1, p=prob, size=len(cart))

    connected_nodes_b = np.unique(
        [y for x, y in np.array(cart)[np.argwhere(mask)].squeeze()]
    )

    n = 0
    while len(set_b) != len(connected_nodes_b):  # prevent disconnected graph
        mask = rng.binomial(n=1, p=prob + (0.01 * n), size=len(cart))
        connected_nodes_b = np.unique(
            [y for x, y in np.array(cart)[np.argwhere(mask)].squeeze()]
        )
        n += 1

    for i, e in enumerate(cart):
        if mask[i]:
            u, v = e
            adj_matrix[u, v] = 1.

    adj_matrix[0, :max(set_a)+1] = 1.
    adj_matrix[min(set_b):, -1] = 1.

    if not directed:
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    assert nx.is_connected(nx.from_numpy_matrix(adj_matrix))

    return adj_matrix
