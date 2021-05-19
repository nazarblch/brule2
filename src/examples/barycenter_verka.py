import sys, os

import torch

from barycenters.simplex import MaxCliq, CliqSampler
from models.autoencoder import StyleGanAutoEncoder

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
heatmapper = ToGaussHeatMap(256, 4)

# [51, 52, 85, 86]
# [1, 44, 34, 94, 10, 72]
# [20, 68, 96, 88]
# [7, 17, 65, 42, 84]
# [71, 72, 73]
# [0, 75, 54, 48, 37]
# [13, 77, 32, 38, 99]
# [9, 40, 36, 66, 67, 78]
# [3, 35, 18, 55, 63, 25, 15]
data_img = []
data_lm = []

def plot_img_with_lm(img: torch.Tensor, lm: torch.Tensor, nrows=4, ncols=4):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows * 2.5, ncols * 2.5))

    for i in range(nrows):
        for j in range(ncols):
            index = i * nrows + j
            axes[i, j].axis("off")
            axes[i, j].imshow(((img[index]).permute(1, 2, 0).numpy() + 1) * 0.5)
            b = lm[index].numpy() * 255
            axes[i, j].scatter(b[:, 0], b[:, 1], c="white", s=2)

    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()




def add_bc(sample):

    landmarks = [ls[i] for i in sample]
    img = torch.cat([images[i][None, ] for i in sample]).cuda()
    B, I, Bws = bc_sampler.sample(landmarks, img)
    I = I.type(torch.float32)

    for i in sample:

        data_img.append(images[i][None, ])
        data_lm.append(torch.from_numpy(ls[i])[None,])

    data_img.append(I[None, ])
    data_lm.append(torch.from_numpy(B)[None,].type(torch.float32))

add_bc([34, 94, 10])
add_bc([7, 17, 65])
add_bc([54, 48, 37])
add_bc([66, 67, 78])

plot_img_with_lm(torch.cat(data_img), torch.cat(data_lm))

