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
from barycenters.sampler import Uniform2DBarycenterSampler, Uniform2DAverageSampler, ImageBarycenterSampler
from parameters.path import Paths
from joblib import Parallel, delayed


N = 100
dataset = LazyLoader.w300().dataset_train
D = np.load(f"{Paths.default.models()}/w300graph{N}.npy")
padding = 68
prob = np.ones(padding) / padding
NS = 7000


def LS(k):
    return dataset[k]["meta"]['keypts_normalized'].numpy()


ls = np.asarray([LS(k) for k in range(N)])
images = [dataset[k]["data"] for k in range(N)]

# bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=1.0)
bc_sampler = ImageBarycenterSampler(padding, dir_alpha=2.0)
heatmapper = ToGaussHeatMap(256, 3)

# [51, 52, 85, 86]
# [1, 44, 34, 94, 10, 72]
# [20, 68, 96, 88]
# [7, 17, 65, 42, 84]
# [71, 72, 73]
# [0, 75, 54, 48, 37]
# [13, 77, 32, 38, 99]
# [9, 40, 36, 66, 67, 78]
# [3, 35, 18, 55, 63, 25, 15]
sample = [9, 40, 36, 66, 82]

landmarks = [ls[i] for i in sample]
img = torch.cat([images[i][None, ] for i in sample]).cuda()
B, I, Bws = bc_sampler.sample(landmarks, img)

for i in sample:

    hm = heatmapper.forward(torch.from_numpy(ls[i])[None, ])[0].sum(0, keepdim=True)
    plt.imshow(((images[i]+hm).permute(1, 2, 0).numpy() + 1) * 0.5)
    plt.show()


hm = heatmapper.forward(torch.from_numpy(B)[None, ])[0].sum(0, keepdim=True)
plt.imshow(((I + hm).permute(1, 2, 0).numpy() + 1) * 0.5)
plt.show()

print(Bws.tolist())
