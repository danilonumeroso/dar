import numpy as np
import typer
import os

from config.data import DATA_SETTINGS
from functools import partial
from math import log
from norm.io import dump
from numpy.random import default_rng
from numpy.typing import NDArray
from pathlib import Path
from utils.data import algorithms
from utils.data.graphs import erdos_renyi_full, two_community, bipartite
from utils.types import Algorithm


def bfs_init(adj, rng, **kwargs):
    num_nodes = adj.shape[0]
    source = rng.choice(num_nodes)

    return adj, source


def ford_fulkerson_init(adj, rng, **kwargs):
    num_nodes = adj.shape[0]

    if kwargs['random_st']:
        source = rng.choice(num_nodes // 2)
        target = rng.choice(range(num_nodes // 2 + 1, num_nodes))

        if source == target:
            target = (source + 1) % num_nodes
    else:
        source, target = 0, num_nodes - 1

    if kwargs['capacity']:
        high = 10
        capacity: NDArray = rng.integers(low=1, high=high, size=(num_nodes, num_nodes)) / high
    else:
        capacity: NDArray = np.ones((num_nodes, num_nodes))

    capacity = np.maximum(capacity, capacity.T) * adj
    capacity = capacity * np.abs((np.eye(num_nodes) - 1))

    return capacity, source, target


_INITS = {
    Algorithm.ff: ford_fulkerson_init,
    Algorithm.ffmc: ford_fulkerson_init
}

_GRAPH_DISTRIB = {
    'two_community': two_community,
    'erdos_renyi': erdos_renyi_full,
    'bipartite': bipartite
}


def accum_samples(alg, params, data):
    algorithm_fn = getattr(algorithms, alg)
    if alg == Algorithm.ek_bfs:
        for _, probes in algorithm_fn(*params):
            data.append(probes)

    else:
        _, probes = algorithm_fn(*params)
        data.append(probes)

    return data


def main(alg: Algorithm,
         dataset_name: str = 'default',
         graph_density: float = 0.35,
         outer_prob: float = 0.05,
         save_path: Path = './data/clrs',
         graph_distrib: str = 'two_community',
         weighted: bool = False,
         directed: bool = False,
         seed: int = None):

    if seed is None:
        seed = int.from_bytes(os.urandom(2), byteorder="big")

    assert graph_distrib in ['two_community', 'erdos_renyi', 'bipartite']

    distrib = _GRAPH_DISTRIB[graph_distrib]

    rng = default_rng(seed)
    init_fn = _INITS[alg]

    save_path = save_path / alg.value / dataset_name

    probs = {}
    graphs = {}
    extras = {}

    # First sample graphs (aids reproducibility).
    for split in DATA_SETTINGS.keys():
        num_nodes = DATA_SETTINGS[split]['length']
        probs[split] = max(graph_density, 1.25*log(num_nodes)/num_nodes)

        distrib = partial(distrib, outer_prob=outer_prob)
        graphs[split] = []
        for _ in range(DATA_SETTINGS[split]['num_samples']):
            adj = distrib(num_nodes=num_nodes,
                          prob=probs[split] if graph_distrib != 'bipartite' else rng.uniform(low=graph_density, high=1),
                          directed=directed,
                          weighted=weighted,
                          rng=rng)
            graphs[split].append(adj)

    # Then run the algorithm for each of them.
    for split in DATA_SETTINGS.keys():
        extras[split] = dict()
        data = []
        num_nodes = DATA_SETTINGS[split]['length']
        with typer.progressbar(range(DATA_SETTINGS[split]['num_samples']), label=split) as progress:
            for i in progress:
                params = init_fn(graphs[split][i],
                                 rng,
                                 random_st=graph_distrib != 'bipartite',
                                 capacity=graph_distrib != 'bipartite')
                data = accum_samples(alg, params, data)

        key = list(data[0]['hint']['node'].keys())[0]
        avg_length = []
        for d in data:
            avg_length.append(d['hint']['node'][key]['data'].shape[0])
        # print statistics

        extras[split]['max'] = max(avg_length)
        extras[split]['avg'] = sum(avg_length) / len(avg_length)
        print("[avg] traj len:", extras[split]['avg'])
        print("[max] traj len:", extras[split]['max'])

        dump(data, save_path / f'{split}_{alg}.pkl')

    dump(dict(seed=seed,
              graph_density=probs,
              **extras),
         save_path / 'config.json')


if __name__ == '__main__':
    typer.run(main)
