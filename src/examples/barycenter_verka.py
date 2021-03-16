import sys, os

import torch

from barycenters.simplex import MaxCliq, CliqSampler

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src/'))
from torchvision.utils import make_grid
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

ls = []
images = []

for k in range(N):
    dk = dataset[k]
    ls.append(dk["meta"]['keypts_normalized'].numpy())
    images.append(dk["data"])


# bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=1.0)
bc_sampler = ImageBarycenterSampler(padding, dir_alpha=4.0)
heatmapper = ToGaussHeatMap(256, 2)

# [51, 52, 85, 86]
# [1, 44, 34, 94, 10, 72]
# [20, 68, 96, 88]
# [7, 17, 65, 42, 84]
# [71, 72, 73]
# [0, 75, 54, 48, 37]
# [13, 77, 32, 38, 99]
# [9, 40, 36, 66, 67, 78]
# [3, 35, 18, 55, 63, 25, 15]
data = []

def add_bc(sample):

    landmarks = [ls[i] for i in sample]
    img = torch.cat([images[i][None, ] for i in sample]).cuda()
    B, I, Bws = bc_sampler.sample(landmarks, img)
    I = I.type(torch.float32)



    for i in sample:

        hm = heatmapper.forward(torch.from_numpy(ls[i])[None, ])[0].sum(0, keepdim=True)
        data.append((hm)[None,])
        # plt.imshow(((images[i]+hm).permute(1, 2, 0).numpy() + 1) * 0.5)
        # plt.show()

    hm = heatmapper.forward(torch.from_numpy(B)[None, ])[0].sum(0, keepdim=True).type(torch.float32)
    plt.imshow(((I + hm).permute(1, 2, 0).numpy() + 1) * 0.5)
    plt.show()

    data.append((hm)[None,])

add_bc([34, 94, 10])
add_bc([7, 17, 65])
add_bc([54, 48, 37])
add_bc([66, 67, 78])

data = torch.cat(data)
grid = make_grid(
    data, nrow=4, padding=2, pad_value=0, normalize=True, range=(-1, 1),
    scale_each=False)

print(grid.shape)
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.show()

