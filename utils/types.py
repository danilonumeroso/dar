from enum import Enum


class Algorithm(str, Enum):
    bfs = 'bfs'
    ek_bfs = 'ek_bfs'
    dijkstra = 'dijkstra'
    astar = 'a_star'
    ff = 'ford_fulkerson'
    ffmc = 'ford_fulkerson_mincut'


class SearchType(str, Enum):
    large = 'large'
    small = 'small'
    one = 'one'


class ModelType(str, Enum):
    dual_edp = 'dual_edp'
    uniform = 'uniform'
    normal = 'normal'
