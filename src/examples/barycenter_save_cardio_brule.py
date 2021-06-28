import sys, os

import torch

from barycenters.simplex import MaxCliq, CliqSampler, AllCliq, UniformCliqSampler
from barycenters.smote import Oversampling

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


N = 300
dataset = LazyLoader.cardio().dataset_train
padding = 200
prob = np.ones(padding) / padding

data_ids = np.random.permutation(np.arange(0, 700))[0: N]

def LS(k):
    return dataset[k]['keypoints'].numpy()

ls = [LS(k) for k in data_ids]

bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=1.0)

bc = bc_sampler.mean(ls)

plt.scatter(bc[:, 0], bc[:, 1])
plt.show()

heatmapper = ToGaussHeatMap(256, 4)

hm = heatmapper.forward(torch.from_numpy(bc)[None,])[0].sum(0)

np.save(f"{Paths.default.models()}/cardio_200_coord.npy", bc)
plt.imshow(hm.numpy())
plt.show()
