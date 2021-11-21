from joblib import Parallel, delayed

import math
import sys, os


from dataset.d300w import ThreeHundredW
import ot

from dataset.lazy_loader import Cardio, LazyLoader
from parameters.path import Paths
import numpy as np
from torch.utils.data import Subset

image_size = 256
padding = 200
N = 300
prob = np.ones(padding) / padding

dataset_train = LazyLoader.cardio_landmarks("cardio_300/lm").dataset_train



def load_landmarks(k):
    return dataset_train[k].numpy()


landmarks = [load_landmarks(k) for k in range(N)]


def compute_w2(i, j):
    M_ij = ot.dist(landmarks[i], landmarks[j])
    D_ij = ot.emd2(prob, prob, M_ij)
    return D_ij


D = np.zeros((N, N))


for i in range(N):
    print(i)
    D[i, i + 1:] = np.array(Parallel(n_jobs=30)(delayed(compute_w2)(i, j) for j in range(i + 1, N)))

D = np.sqrt(D + D.T)

np.save(f"{Paths.default.models()}/cardio_graph{N}.npy", D)






