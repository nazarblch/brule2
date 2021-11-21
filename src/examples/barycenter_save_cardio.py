import sys, os
import time

import torch

from barycenters.simplex import MaxCliq, CliqSampler


from dataset.lazy_loader import LazyLoader, W300DatasetLoader, CelebaWithKeyPoints, Celeba
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from dataset.toheatmap import ToGaussHeatMap
from dataset.probmeasure import UniformMeasure2D01
import pandas as pd
import networkx as nx
import ot
from barycenters.sampler import Uniform2DBarycenterSampler, Uniform2DAverageSampler
from parameters.path import Paths
from joblib import Parallel, delayed

N = 701
dataset = LazyLoader.cardio_landmarks(f"cardio_{N}/lm").dataset_train
D = np.load(f"{Paths.default.models()}/cardio_graph{N}.npy")
padding = 200
prob = np.ones(padding) / padding
NS = 1000


def LS(k):
    return dataset[k].numpy()


ls = np.asarray([LS(k) for k in range(N)])

def viz_mes(ms):
    heatmaper = ToGaussHeatMap(128, 1)

    kapusta = torch.zeros(128, 128)
    for m in ms:
        keyptsiki = torch.from_numpy(m)[None,].clamp(0, 1)
        tmp = heatmaper.forward(keyptsiki)
        kapusta += tmp.sum(axis=(0, 1))

    plt.imshow(kapusta)
    plt.show()

    return kapusta / kapusta.sum()

ls_mes = viz_mes(ls)

bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=1.0)
# bc_sampler = Uniform2DAverageSampler(padding, dir_alpha=1.0)

tt = time.time()

def juja(a, b):

    def juja_inside(sample):
        landmarks = [ls[i] for i in sample]
        B, Bws = bc_sampler.sample(landmarks)
        print(sample)
        return B

    cliques, K = MaxCliq(a, b).forward(D)
    cl_sampler = CliqSampler(cliques)
    cl_samples = cl_sampler.sample(NS)

    bc_samples = list(Parallel(n_jobs=30)(delayed(juja_inside)(sample) for sample in cl_samples))

    # bc_mes = viz_mes(bc_samples)
    # ent = kl(ls_mes, bc_mes) + kl(bc_mes, ls_mes)

    return None, bc_samples


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


ent, bcs = juja(a=0.2, b=9)
print(ent)
print(tt - time.time())

# os.mkdir(f"{Paths.default.data()}/w300_bc_{N}_avg")
# os.mkdir(f"{Paths.default.data()}/w300_bc_{N}_avg/lmbc")
# os.mkdir(f"{Paths.default.data()}/w300_bc_{N}_avg/lm")
#
# for i,b in enumerate(bcs):
#     np.save(f"{Paths.default.data()}/cardio_{N}/lmbc/{i}.npy", b)
#
# for i,b in enumerate(ls):
#     np.save(f"{Paths.default.data()}/cardio/lm/{i}.npy", b)

# ent = kl(ls_mes, bc_mes) + kl(bc_mes, ls_mes)
#
# print(ent)
#