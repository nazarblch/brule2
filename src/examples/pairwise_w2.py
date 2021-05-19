from joblib import Parallel, delayed

import math
import sys, os
from tqdm import tqdm


sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/gans/'))


from dataset.d300w import ThreeHundredW
import ot
from parameters.path import Paths
import numpy as np
from torch.utils.data import Subset

image_size = 256
padding = 68
N = 300
prob = np.ones(padding) / padding

print("N", N)

dataset_train = Subset(ThreeHundredW(f"{Paths.default.data()}/300w", train=True, imwidth=500, crop=15), range(N))


def load_landmarks(k):
    return dataset_train[k]['meta']['keypts_normalized'].numpy()


landmarks = np.asarray([load_landmarks(k) for k in range(N)])


def compute_w2(i, j):

    M_ij = ot.dist(landmarks[i], landmarks[j])
    D_ij = ot.emd2(prob, prob, M_ij)
    return D_ij


D = np.zeros((N, N))


for i in tqdm(range(N)):
    D[i, i+1:] = np.array(Parallel(n_jobs=30)(delayed(compute_w2)(i, j) for j in range(i+1, N)))

D = np.sqrt(D + D.T)

np.save(f"{Paths.default.models()}/w300graph{N}.npy", D)






