from joblib import Parallel, delayed

import math
import sys, os

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/gans/'))

from matplotlib import pyplot as plt
from dataset.d300w import ThreeHundredW
import ot
from barycenters.gmm import MaskToMixtureGPU, MaskToMixture
from dataset.brats import FilterDataset, Dataset3DTo2D, BraTS3D
from dataset.hum36 import SimpleHuman36mDataset
from parameters.dataset import DatasetParameters
from dataset.toheatmap import ToGaussHeatMap
from dataset.lazy_loader import Cardio, BraTSLoader
from parameters.path import Paths
import numpy as np
from torch.utils.data import Subset



data = BraTSLoader().dataset_train
print(data.__len__())

padding = 200
N = 3529
mask_to_mes = MaskToMixture(256, padding)
heatmapper = ToGaussHeatMap(256, 1)

dataset_train = Subset(data, range(N))


def load_landmarks(k):
    print(k)
    tt, seg = dataset_train[k]
    g = mask_to_mes.forward(seg, 0.0001)

    return g.coord[0].numpy()

landmarks = np.asarray(Parallel(n_jobs=40)(
    delayed(load_landmarks)(k) for k in range(N)
), dtype=object)


# os.mkdir(f"{Paths.default.data()}/brats_lm")
for i in range(N):
    np.save(f"{Paths.default.data()}/brats_lm/{i}.npy", landmarks[i])


landmarks = np.zeros((N, padding, 2))
lens = np.zeros(N, dtype=np.int)
prob = np.zeros((N, padding))

for i in range(N):
    tmp = np.load(f"{Paths.default.data()}/brats_lm/{i}.npy")
    lens[i] = tmp.shape[0]
    landmarks[i, 0: lens[i]] = tmp
    prob[i, 0: lens[i]] = np.ones(tmp.shape[0]) / tmp.shape[0]

print(landmarks.shape)

def compute_w2(i, j):
    lmi = landmarks[i, 0: lens[i]]
    lmj = landmarks[j, 0: lens[j]]
    M_ij = ot.dist(lmi, lmj)
    pi = prob[i, 0: lens[i]]
    pj = prob[j, 0: lens[j]]
    D_ij = ot.emd2(pi, pj, M_ij)
    return D_ij


D = np.zeros((N, N))


for i in range(N):
    print(i)
    D[i, i + 1:] = np.array(Parallel(n_jobs=40)(delayed(compute_w2)(i, j) for j in range(i + 1, N)))

D = np.sqrt(D + D.T)

np.save(f"{Paths.default.models()}/brats_graph{N}.npy", D)






