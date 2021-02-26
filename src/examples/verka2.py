# %%

import argparse
import sys, os

from barycenters.sampler import Uniform2DBarycenterSampler
from parameters.path import Paths

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/gans_pytorch/gan/'))

from dataset.lazy_loader import LazyLoader, W300DatasetLoader, CelebaWithKeyPoints, Celeba

# %%

from sklearn.neighbors import NearestNeighbors


# %%

import ot
import numpy as np
from tqdm import tqdm_notebook as tqdm

# %%

from joblib import Parallel, delayed

# %%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



from dataset.toheatmap import ToGaussHeatMap
from dataset.probmeasure import UniformMeasure2D01

heatmaper = ToGaussHeatMap(256, 4)

# %%

W300DatasetLoader.batch_size = 1

# %%

import torch

# %%

kapusta = torch.zeros(256, 256)
for i in range(10):
    w300_train = next(iter(LazyLoader.cardio().loader_train_inf))
    keyptsiki = w300_train['keypoints']
    tmp = heatmaper.forward(keyptsiki)
    kapusta += tmp.sum(axis=(0, 1))


plt.imshow(kapusta)
plt.show()

padding = 200


def LS(k):
    return LazyLoader.cardio().dataset_train[k]['keypoints'].numpy()

ls = [LS(k) for k in range(100)]

D = np.load(f"{Paths.default.models()}/cardio_graph{100}.npy")

# %%

nbrs = NearestNeighbors(n_neighbors=10, metric="precomputed").fit(D)
distances, indices = nbrs.kneighbors(D)

# %%

X_init = np.random.uniform(size=(padding, 2))

# %%

weights = np.array([0.1] * 10)


# %%

sampler = Uniform2DBarycenterSampler(padding)

# %%

for ii, idx in enumerate(indices):
    B = sampler.sample([
        ls[idx[m]] for m in range(10)
    ])

    fig, ax = plt.subplots(nrows=1, ncols=11, figsize=(16.75, 1.25))

    for k in range(10):
        ax[k + 1].invert_yaxis()
        ax[k + 1].set_title("d: {:.3f}".format(distances[ii][k]))
        ax[k + 1].scatter(ls[idx[k]][:, 0], ls[idx[k]][:, 1], c="r", s=7)

    ax[0].invert_yaxis()
    ax[0].set_title("B {}".format(ii))
    ax[0].scatter(B[:, 0], B[:, 1], c="b", s=7)

    plt.show()


plt.hist(D.reshape(-1))
plt.show()

# %%

import pandas as pd
import networkx as nx

#
# # %%
#
# def threshold(A, eps=0.3):
#     B = A.copy()
#     B = (B < eps).astype(int)
#     np.fill_diagonal(B, 0)
#     return B
#
#
# def knn(A, k=5):
#     nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed").fit(A)
#     K = nbrs.kneighbors_graph(A).toarray().astype(int)
#     np.fill_diagonal(K, 0)
#     return K + K.T
#
#
# # %%
#
# def _knn(A, k=3):
#     B = A.copy()
#
#     for i, _ in enumerate(B):
#         knn_dist_i = np.sort(B[i])[k - 1]
#         B[i] = (B[i] <= knn_dist_i).astype(int)
#
#     np.fill_diagonal(B, 0)
#     return B + B.T
#
#
# # %%
#
# A = threshold(D, 0.04)
# K = knn(D, 10)
#
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 5.5))
# ax[0].imshow(A)
# ax[1].imshow(K)
# ax[2].imshow(D)
# plt.show()
#
# # %%
#
# G = nx.from_numpy_array(K)
#
# # %%
#
# simplices = []
#
# min_clique, max_clique = 2, 20
#
# cliques = nx.find_cliques(G)
# max_clique_size = 0
#
# for clique in cliques:
#     if (len(clique) >= min_clique) & (len(clique) <= max_clique):
#         simplices.append(clique)
#
#         if len(clique) > max_clique_size:
#             max_clique_size = len(clique)
#
#
# # %%
#
#
#
# # %%
#
# i = 13
# fig, ax = plt.subplots(nrows=1, ncols=len(simplices[i]), figsize=(16, 2))
# for k in range(len(simplices[i])):
#     ax[k].invert_yaxis()
#     ax[k].scatter(LS(simplices[i][k])[:, 0], LS(simplices[i][k])[:, 1], c="r", s=7)
# plt.show()
#

# %%


# %%


# %%


# %%


# %%


# %%


# %%



