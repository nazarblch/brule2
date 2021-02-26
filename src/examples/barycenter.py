import sys, os

import torch

from barycenters.simplex import MaxCliq, CliqSampler

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/'))

from dataset.lazy_loader import LazyLoader, W300DatasetLoader, CelebaWithKeyPoints, Celeba
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from dataset.toheatmap import ToGaussHeatMap
from dataset.probmeasure import UniformMeasure2D01
import pandas as pd
import networkx as nx
import ot
from barycenters.sampler import Uniform2DBarycenterSampler
from parameters.path import Paths
from joblib import Parallel, delayed



N = 300
dataset = LazyLoader.w300().dataset_train
D = np.load(f"{Paths.default.models()}/w300graph{N}.npy")
padding = 68
prob = np.ones(padding) / padding


def LS(k):
    return dataset[k]["meta"]['keypts_normalized'].numpy()


ls = np.asarray([LS(k) for k in range(N)])

def viz_mes(ms):
    heatmaper = ToGaussHeatMap(128, 1)

    kapusta = torch.zeros(128, 128)
    for m in ms:
        keyptsiki = torch.from_numpy(m)[None,]
        tmp = heatmaper.forward(keyptsiki)
        kapusta += tmp.sum(axis=(0, 1))

    plt.imshow(kapusta)
    plt.show()

    return kapusta / kapusta.sum()

ls2 = np.asarray([LS(k) for k in range(N, 3*N)])
ls_mes = viz_mes(ls2)


def compute_w2(l1, l2):

    M_ij = ot.dist(l1, l2)
    D_ij = ot.emd2(prob, prob, M_ij)
    return D_ij

bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=0.4)

parameter_a = np.arange(0.05, 0.15, 0.01)
parameter_b = range(3, 15, 1)


def juja(a, b):

    def juja_inside(sample):
        landmarks = [ls[i] for i in sample]
        B, Bws = bc_sampler.sample(landmarks)
        return B

    cliques, K = MaxCliq(a, b).forward(D)
    cl_sampler = CliqSampler(cliques)
    cl_samples = cl_sampler.sample(2 * N)

    bc_samples = list(Parallel(n_jobs=30)(delayed(juja_inside)(sample) for sample in cl_samples))

    bc_mes = viz_mes(bc_samples)
    ent = kl(ls_mes, bc_mes) + kl(bc_mes, ls_mes)

    return ent


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


best = 10000

for a in parameter_a:

    suma = 0

    for b in parameter_b:
        res = juja(a, b)
        suma += res / len(parameter_b)

        print("a: {}, b: {}, res: {}".format(a, b, res))

    print(a, suma)



# ent = kl(ls_mes, bc_mes) + kl(bc_mes, ls_mes)
#
# print(ent)
#