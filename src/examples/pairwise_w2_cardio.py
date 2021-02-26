from joblib import Parallel, delayed

import math
import sys, os


sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/gans/'))


from dataset.d300w import ThreeHundredW
import ot

from dataset.lazy_loader import Cardio
from parameters.path import Paths
import numpy as np
from torch.utils.data import Subset

image_size = 256
padding = 200
N = 701
prob = np.ones(padding) / padding

dataset_train = Subset(Cardio().dataset_train, range(N))


def load_landmarks(k):
    return dataset_train[k]["keypoints"].numpy()


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






