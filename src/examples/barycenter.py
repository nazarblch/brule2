import multiprocessing
import sys, os

from tqdm import tqdm

from barycenters.simplex import MaxCliq, CliqSampler, AllCliq
from gan.loss.hinge import HingeLoss
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from gan.nn.stylegan.components import EqualLinear, EqualConv2d
from nn.common.view import View
from stylegan2_bk.model import EqualConv2d

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
from torch import nn


N = 300
dataset = LazyLoader.w300().dataset_train

padding = 68
prob = np.ones(padding) / padding




def LS(k):
    return dataset[k]["meta"]['keypts_normalized'].numpy()


data_ids = np.random.permutation(np.arange(0, 3148))

ls = np.asarray([LS(k) for k in data_ids[0: N]])

ls2 = np.asarray([LS(k) for k in data_ids[N: 2*N]])


def compute_w22(i, j):

    M_ij = ot.dist(ls[i], ls[j])
    D_ij = ot.emd2(prob, prob, M_ij)
    return D_ij


D = np.zeros((N, N))


for i in range(N):
    D[i, i+1:] = np.array(Parallel(n_jobs=30)(delayed(compute_w22)(i, j) for j in range(i+1, N)))

D = np.sqrt(D + D.T)



print("D matrix")


def compute_w2(l1, l2):
    M_ij = ot.dist(l1, l2)
    D_ij = ot.emd2(prob, prob, M_ij, processes=10)
    return D_ij

bc_sampler = Uniform2DBarycenterSampler(padding, dir_alpha=1.0)


def juja(knn):

    def juja_inside(sample):
        landmarks = [ls[i] for i in sample]
        B, Bws = bc_sampler.sample(landmarks)
        return B

    cliques, K = AllCliq(knn).forward(D)
    cl_sampler = CliqSampler(cliques)
    cl_samples = cl_sampler.sample(N)

    bc_samples = list(Parallel(n_jobs=30)(delayed(juja_inside)(sample) for sample in cl_samples))

    disc_loss = parameters_wloss(np.asarray(bc_samples), ls2)
    print(knn, disc_loss)

    return disc_loss



def parameters_wloss(ldmrks1, ldmrks2):

    D2 = np.zeros((N, N))
    prob_d = np.ones(N) / (N)

    for i in tqdm(range(N)):
        # print(i)
        D2[i, :] = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(compute_w2)(ldmrks1[i], ldmrks2[j]) for j in range(0, N)
        ))

    return ot.emd2(prob_d, prob_d, D2)

res = juja(3)
res = juja(5)
res = juja(7)
res = juja(9)
res = juja(11)


