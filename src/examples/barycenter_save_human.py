import sys, os

import torch

from barycenters.simplex import MaxCliq, CliqSampler
from dataset.hum36 import SimpleHuman36mDataset
from parameters.dataset import DatasetParameters

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
from barycenters.sampler import Uniform2DBarycenterSampler, Uniform2DAverageSampler
from parameters.path import Paths
from joblib import Parallel, delayed

N = 13410
D = np.load(f"{Paths.default.models()}/hum36_graph{N}.npy")
padding = 32
prob = np.ones(padding) / padding
NS = 13000

print(D.reshape(-1).mean())
plt.hist(D.reshape(-1), bins=30)
plt.show()

parser = DatasetParameters()
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

data = SimpleHuman36mDataset()
data.initialize(args.data_path)


def LS(k):
    return data[k]["paired_B"].numpy()


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
bc_sampler_2 = Uniform2DAverageSampler(padding, dir_alpha=1.0)

def compute_w2(li, lj):
    M_ij = ot.dist(li, lj)
    D_ij = ot.emd2(prob, prob, M_ij)
    return D_ij

def draw_lm(lm1, lm2):
    heatmaper = ToGaussHeatMap(128, 1)
    keyptsiki = torch.from_numpy(lm1)[None,]
    tmp1 = heatmaper.forward(keyptsiki).type(torch.float32).sum(1, keepdim=True)
    lm2[:, 0] += 0.1
    keyptsiki = torch.from_numpy(lm2)[None,]
    tmp2 = heatmaper.forward(keyptsiki).type(torch.float32).sum(1, keepdim=True)
    tmp = torch.cat([tmp1, tmp2, torch.zeros_like(tmp2)], dim=1)
    plt.imshow(tmp[0].permute(1, 2, 0).numpy())
    plt.show()

def juja(a, b):

    def juja_inside(sample):
        landmarks = [ls[i] for i in sample]
        cmp = 10
        B2 = None
        while cmp > 0.0001:
            B, Bws = bc_sampler.sample(landmarks)
            B2, _ = bc_sampler_2.sample(landmarks, Bws)
            cmp = compute_w2(B, B2)

        print(Bws)
        # draw_lm(B, B2)
        return B2

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


ent, bcs = juja(a=0.26, b=6)
print(ent)

heatmaper = ToGaussHeatMap(128, 1)
keyptsiki = torch.from_numpy(bcs[1])[None,].clamp(0, 1)
tmp = heatmaper.forward(keyptsiki)
plt.imshow(tmp.sum((0, 1)).numpy())
plt.show()

# os.mkdir(f"{Paths.default.data()}/human_part_{N}")
# os.mkdir(f"{Paths.default.data()}/human_part_{N}/lmbc")
# os.mkdir(f"{Paths.default.data()}/human_part_{N}/lm")

for i,b in enumerate(bcs):
    np.save(f"{Paths.default.data()}/human_part_{N}/lmbc/{i}.npy", b)

for i,b in enumerate(ls):
    np.save(f"{Paths.default.data()}/human_part_{N}/lm/{i}.npy", b)

