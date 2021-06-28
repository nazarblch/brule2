import numpy as np
import pandas as pd
import numbers
from itertools import combinations
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict
from collections import defaultdict

K = {
    "knn": lambda i, j, D, knn_distance, d: int((D[i, j] / max(knn_distance[i], knn_distance[j])) < 1),
    "cknn": lambda i, j, D, knn_distance, d: int((D[i, j] / (d * np.sqrt(knn_distance[i] * knn_distance[j]))) < 1)
}

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def threshold(A, eps1=0.0, eps2=0.3):
    B = A.copy()
    B = (B < eps2).astype(int) * (B > eps1).astype(int)
    np.fill_diagonal(B, 0)
    return B


def knn(A, k=5):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(A)
    K = nbrs.kneighbors_graph(A).toarray().astype(int)
    np.fill_diagonal(K, 0)
    return K + K.T


class MaxCliq:

    def __init__(self, threshold_b=0.1, nn=10, threshold_l=0.0):
        self.threshold_b = threshold_b
        self.nn = nn
        self.threshold_l = threshold_l

    def forward(self, matrix: np.ndarray):
        assert matrix.shape.__len__() == 2
        assert matrix.shape[0] == matrix.shape[1]
        # K = threshold(matrix, self.threshold_b)
        K = knn(matrix, self.nn)
        K1 = threshold(matrix, self.threshold_l, self.threshold_b)
        K = K * K1
        G = nx.from_numpy_array(K)
        min_clique, max_clique = 2, 20
        cliques = nx.find_cliques(G)
        cliques = list(filter(lambda c: max_clique > len(c) >= min_clique, cliques))
        return cliques, K


class AllCliq:

    def __init__(self, nn=10):
        self.nn = nn
        self.n_jobs = 20

    def forward(self, matrix: np.ndarray):

        k = self.nn
        D = matrix
        k_ambient = 3

        # find distances to k-neighbors
        distances, _ = NearestNeighbors(n_neighbors=k + 1, metric="precomputed", n_jobs=self.n_jobs)\
            .fit(D).kneighbors(D)
        knn_distance = distances[:, k]  # distance to k(=7)-neighbor (local density estimator)

        # create weighted adjacency matrix of a graph, with weights given by appropriate kernel
        A = np.zeros_like(D)

        for i in range(D.shape[0]):
            for j in range(D.shape[0]):
                A[i, j] = K["knn"](i, j, D, knn_distance, 1.0)

        np.fill_diagonal(A, 0)

        # create graph
        G = nx.from_numpy_array(A)

        # points positions in dictionary of tuples format
        # self.pos = {i: (x[0], x[1]) for i, x in enumerate(X)}

        # find maximal simplices (of dimension n)
        maximal_simplices = list(nx.find_cliques(G))

        # enumerate faces of max dimension d in maximal simplicies
        simplices = []
        for maximal_simplex in maximal_simplices:
            k_maximal_simplex = len(maximal_simplex)

            k_simplices = list(combinations(maximal_simplex, min(k_ambient, k_maximal_simplex)))
            for k_simplex in k_simplices:
                simplices.append(sorted(list(k_simplex)))

        return simplices, A


class CliqSampler:

    def __init__(self, cliques: List[np.ndarray]):

        self.nodes: Dict[float] = defaultdict(lambda: 0.0)
        self.cliques = np.asarray(cliques)

        for c in cliques:
            for v in c:
                self.nodes[v] += 1.0

        for v in self.nodes.keys():
            self.nodes[v] = 1 / self.nodes[v]

        self.prob = np.zeros(len(cliques))

        for i, c in enumerate(cliques):
            for v in c:
                self.prob[i] += self.nodes[v]

        self.prob = self.prob / self.prob.sum()

    def sample(self, number):
        return np.random.choice(self.cliques, number, p=self.prob)


class UniformCliqSampler:

    def __init__(self, cliques: List[np.ndarray]):
        self.cliques = np.asarray(cliques)

    def sample(self, number, power=None):

        if power == None:
            dimensions = np.ones(self.cliques.shape[0])
        else:
            dimensions = np.array([len(item) for item in self.cliques])  # ** power

        # compute probability proportional to simplex dimension
        dimension_p = dimensions / dimensions.sum()

        # choose n simplices to sample from
        random_instance = check_random_state(0)
        idx = random_instance.choice(np.arange(0, len(self.cliques)), size=number, replace=True, p=dimension_p)
        simplices_sample = [self.cliques[i] for i in idx]

        return simplices_sample




