from joblib import Parallel, delayed

import math
import sys, os

from dataset.hum36 import SimpleHuman36mDataset
from parameters.dataset import DatasetParameters



from dataset.d300w import ThreeHundredW
import ot

from dataset.lazy_loader import Cardio
from parameters.path import Paths
import numpy as np
from torch.utils.data import Subset

parser = DatasetParameters()
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

data = SimpleHuman36mDataset()
data.initialize(args)

padding = 32
N = 13410
prob = np.ones(padding) / padding

dataset_train = Subset(data, range(N))


def load_landmarks(k):
    return dataset_train[k]["paired_B"].numpy()


landmarks = np.asarray([load_landmarks(k) for k in range(N)])


def compute_w2(i, j):
    M_ij = ot.dist(landmarks[i], landmarks[j])
    D_ij = ot.emd2(prob, prob, M_ij)
    return D_ij

def compute_l2(i, j):
    return ((landmarks[i]-landmarks[j]) ** 2).sum(1).mean()

D = np.zeros((N, N))


for i in range(N):
    print(i)
    D[i, i + 1:] = np.array(Parallel(n_jobs=30)(delayed(compute_l2)(i, j) for j in range(i + 1, N)))

D = np.sqrt(D + D.T)

np.save(f"{Paths.default.models()}/hum36_graph{N}.npy", D)






