import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict
from collections import defaultdict


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




