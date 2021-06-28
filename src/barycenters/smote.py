import numpy as np
import networkx as nx
import numbers

from itertools import combinations
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

import ot

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


def wasserstein_barycenter(X, weights=None, B_init=None, verbose=False):
    # n points in simplex, m points in point cloud
    n, m, d = X.shape

    # if barycentric coordinates are unspecified use random ~Dir
    if weights == None:
        weights = np.random.dirichlet(np.ones(n), size=1)[0]

    # if barycenter support is unspecified use random ~U
    if B_init == None:
        B_init = np.random.uniform(size=(m, d))

    # arrange spanning point clouds as a list
    X_list = []
    for X_i in X:
        X_list.append(X_i)

    # uniform measures for barycenter and point clouds
    b = np.ones(m) / m
    a_list = [b] * n

    B = ot.lp.free_support_barycenter(X_list, a_list, B_init, b, weights, verbose=verbose)

    return B


class Oversampling:

    def __init__(self, X, k=7, kernel="knn", d=1.0, n_jobs=None, random_state=None):

        self.X = X
        k_ambient = X.shape[-1] + 1
        n, m, _ = X.shape

        self.n_jobs = n_jobs
        self.random_state = random_state

        # distance matrix
        D = np.zeros((n, n))

        # uniform measures as points
        a = np.ones(m) / m
        b = np.ones(m) / m

        for i in range(n):
            for j in range(i + 1, n):
                # squared Euclidean distance as the ground metric
                M_ij = ot.dist(X[i], X[j], metric="sqeuclidean")

                # 2-Wasserstein distance, take square root
                D[i, j] = np.sqrt(ot.emd2(a, b, M_ij))

        # symmetrize distance matrix
        self.D = D + D.T

        # find distances to k-neighbors
        distances, _ = NearestNeighbors(n_neighbors=k + 1, metric="precomputed", n_jobs=self.n_jobs).fit(
            self.D).kneighbors(self.D)
        knn_distance = distances[:, k]  # distance to k(=7)-neighbor (local density estimator)

        # create weighted adjacency matrix of a graph, with weights given by appropriate kernel
        A = np.zeros_like(self.D)

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                A[i, j] = K[kernel](i, j, self.D, knn_distance, d)

        np.fill_diagonal(A, 0)

        # create graph
        self.G = nx.from_numpy_array(A)

        # points positions in dictionary of tuples format
        self.pos = {i: (x[0], x[1]) for i, x in enumerate(X)}

        # find edges
        self.edges = []
        for edge in nx.edges(self.G):
            self.edges.append(list(edge))

        # find maximal simplices (of dimension n)
        self.maximal_simplices = list(nx.find_cliques(self.G))

        # enumerate faces of max dimension d in maximal simplicies
        self.simplices = []
        for maximal_simplex in self.maximal_simplices:
            k_maximal_simplex = len(maximal_simplex)

            k_simplices = list(combinations(maximal_simplex, min(k_ambient, k_maximal_simplex)))
            for k_simplex in k_simplices:
                self.simplices.append(sorted(list(k_simplex)))

    def sample(self, method="simplicial", n=(10, 1), alpha=1.0, power=None, with_replacement=True, random=True):

        if method == "SMOTE":
            simplices = self.edges
        elif method == "simplicial":
            simplices = self.simplices
        elif method == "simplicial_maximal":
            simplices = self.maximal_simplices
        else:
            raise ValueError(
                "Unrecognized method of sampling '{0}'. Method should be 'simplicial', 'simplicial_maximal' or 'SMOTE'.".format(
                    method))

        # create points array
        p = []

        if power == None:
            dimensions = np.ones(len(simplices))

        else:
            # find dimension of simplices
            dimensions = np.array([len(item) for item in simplices])  # ** power

        # compute probability proportional to simplex dimension
        dimension_p = dimensions / dimensions.sum()

        # choose n simplices to sample from
        random_instance = check_random_state(self.random_state)
        idx = random_instance.choice(np.arange(0, len(simplices)), size=n[0], replace=with_replacement, p=dimension_p)
        simplices_sample = [simplices[i] for i in idx]  # simplices_sample = simplices

        # set barycenric coordinates
        for simplex in simplices_sample:
            k = len(simplex)

            # sample n points from a simplex
            for _ in range(n[1]):

                # at random
                if random:
                    # sample a point p ~ Dirichlet(alpha) from a simplex
                    # alpha < 1: close to endpoints, alpha = 1: uniform, alpha > 1: close to center
                    B = random_instance.dirichlet(np.ones(k) * alpha, size=1)[0]
                    B = B / B.sum()  # ???

                # or from a simplex center
                else:
                    B = np.ones(k) / k

                # compute Wasserstein barycenter of a new point
                barycenter_point = wasserstein_barycenter(self.X[simplex])  # B @ self.X[simplex]

                p.append(barycenter_point)

        return np.array(p)