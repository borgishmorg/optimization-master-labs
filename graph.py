import numpy as np
import numba as nb
import networkx as nx
import contextlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
from typing import Tuple

# На форме (используете tkinter) задаются в произвольных местах точек
# (количество устанавливается в настройках программы)
# Реализовать алгоритм Флойда поиска кратчайшего пути при помощи
# параллельных вычислений. Измерить время выполнения

SEED = 42
NODES = 100
EDGES = int(NODES * 1.2)
MEAN_WEIGHT = 1
STD_MEAN = 0.3
MIN_WEIGHT = 0.01
MAX_WEIGHT = 2.


@contextlib.contextmanager
def timer(label: str):
    start = time()
    yield
    print(label, f'{time() - start:.3f} s')


@nb.njit(cache=True)
def init_result_and_next_node(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    result = dist.copy()
    next_node = np.full((NODES, NODES), np.inf)
    for i in range(NODES):
        for j in range(NODES):
            if dist[i, j] != np.inf:
                next_node[i, j] = j
    return result, next_node


@nb.njit(cache=True)
def floyd_serial(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    result, next_node = init_result_and_next_node(dist)

    for k in range(NODES):
        for i in range(NODES):
            for j in range(NODES):
                if result[i, k] + result[k, j] < result[i, j]:
                    result[i, j] = result[i, k] + result[k, j]
                    next_node[i, j] = next_node[i, k]

    return result, next_node


@nb.njit(cache=True)
def inner_floyd_loop(
    k: int,
    i: int,
    result: np.ndarray,
    next_node: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    for j in range(NODES):
        if result[i, k] + result[k, j] < result[i, j]:
            result[i, j] = result[i, k] + result[k, j]
            next_node[i, j] = next_node[i, k]


@nb.njit(cache=True)
def floyd_parallel(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    result, next_node = init_result_and_next_node(dist)

    for k in range(NODES):
        for i in nb.prange(NODES):
            inner_floyd_loop(k, i, result, next_node)

    return result, next_node


if __name__ == '__main__':
    np.random.seed(SEED)
    # generate

    dist = np.full((NODES, NODES), np.inf)
    for i in range(NODES):
        dist[i, i] = 0.

    edges = []
    for i in range(EDGES):
        u, v = np.random.randint(
            0, NODES - 1),  np.random.randint(0, NODES - 1)
        while u == v or dist[u, v] != np.inf:
            u, v = np.random.randint(
                0, NODES - 1),  np.random.randint(0, NODES - 1)
        dist[u, v] = max(
            MIN_WEIGHT,
            min(
                np.random.normal(MEAN_WEIGHT, STD_MEAN),
                MAX_WEIGHT,
            ),
        )
        edges.append((u, v, {'weight': dist[u, v]}))

    with timer('SP:'):
        calculated_serial, next_serial = floyd_serial(dist)

    with timer('PP:'):
        calculated_parallel, next_parallel = floyd_parallel(dist)

    assert (calculated_serial == calculated_parallel).all()
    assert (next_serial == next_parallel).all()

    # draw
    G = nx.DiGraph()
    G.add_nodes_from(range(NODES))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=SEED)

    edge_colors = [edge[-1] for edge in G.edges.data('weight')]
    edge_alphas = [0.3 + 0.7 * edge[-1] /
                   MAX_WEIGHT for edge in G.edges.data('weight')]
    cmap = plt.cm.copper_r

    nx.draw_networkx_nodes(
        G,
        pos,
    )
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )
    calculated_serial[np.isinf(calculated_serial)] = -1
    start, end = np.unravel_index(
        np.argmax(calculated_serial),
        calculated_serial.shape
    )
    path = [start]
    while int(next := next_serial[start, end]) != end:
        next = int(next)
        path.append(next)
        start = next
    path.append(end)
    print(
        f'Max dist from {path[0]} to {end} ({calculated_serial[start, end]:.3f})'
    )
    print(f'{len(path)-1} hops:', ' -> '.join(str(n) for n in path))

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(nx.utils.pairwise(path)),
        arrowstyle="->",
        arrowsize=15,
        edge_color='red',
        width=3,
    )
    nx.draw_networkx_labels(G, pos)

    # set alpha value for each edge
    for i in range(EDGES):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.show()
