import sys, os

import torch

from barycenters.simplex import MaxCliq, CliqSampler
from dataset.cfinder import ContFinder, fill_contour
from dataset.hum36 import SimpleHuman36mDataset
from parameters.dataset import DatasetParameters

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/'))

from dataset.lazy_loader import LazyLoader, W300DatasetLoader, CelebaWithKeyPoints, Celeba, BraTSLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from dataset.toheatmap import ToGaussHeatMap
from dataset.probmeasure import UniformMeasure2D01
import pandas as pd
import networkx as nx
import ot
from barycenters.sampler import Uniform2DBarycenterSampler, Uniform2DAverageSampler, SegmentationBarycenterSampler
from parameters.path import Paths
from joblib import Parallel, delayed

N = 3529
D = np.load(f"{Paths.default.models()}/brats_graph{N}.npy")
NS = 1000

print(D.reshape(-1).mean())
plt.hist(D.reshape(-1), bins=30)
plt.show()

data = BraTSLoader().dataset_train

def LS(k):
    cont = ContFinder().get_contours(data[k][1].numpy()) / 255
    return cont

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

bc_sampler = Uniform2DBarycenterSampler(500, dir_alpha=1.0)

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

    bc_mes = viz_mes(bc_samples)
    ent = kl(ls_mes, bc_mes) + kl(bc_mes, ls_mes)

    return ent, bc_samples


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


ent, bcs = juja(a=0.20, b=6)
print(ent)

bcs_seg = []

for b in bcs:
    seg = fill_contour(b)[np.newaxis, ]
    bcs_seg.append(seg)

os.mkdir(f"{Paths.default.data()}/brats_lm_{N}")
os.mkdir(f"{Paths.default.data()}/brats_lm_{N}/lmbc")
os.mkdir(f"{Paths.default.data()}/brats_lm_{N}/lm")


for i,b in enumerate(bcs_seg):
    np.save(f"{Paths.default.data()}/brats_lm_{N}/lmbc/{i}.npy", b)

# for i,b in enumerate(ls):
#     np.save(f"{Paths.default.data()}/brats_lm_{N}/lm/{i}.npy", b)

