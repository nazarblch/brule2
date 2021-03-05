import multiprocessing

import numpy as np
import torch.nn as nn
import torch
from joblib import Parallel, delayed
from typing import Callable

from dataset.probmeasure import ProbabilityMeasure
from geomloss import SamplesLoss
import ot
from gan.loss.loss_base import Loss
from torch import Tensor


class WeightedSamplesLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, blur=.01, scaling=.9, diameter=None, p: int = 2):
        super(WeightedSamplesLoss, self).__init__()
        self.weights = weights
        self.loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=False, diameter=diameter, p=p)

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):
        batch_loss = self.loss(m1.probability, m1.coord, m2.probability, m2.coord)
        return Loss((batch_loss * self.weights).sum())


class WasLoss(nn.Module):
    def __init__(self, blur=.01, scaling=.9, diameter=None, p: int = 2):
        super(WasLoss, self).__init__()
        self.loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=False, diameter=diameter, p=p)

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):
        batch_loss = self.loss(m1.probability, m1.coord, m2.probability, m2.coord)
        return Loss(batch_loss.mean())


def compute_ot_matrix(x1, x2):
    M = ot.dist(x1, x2, metric='euclidean')
    a = np.ones(x1.shape[0]) / x1.shape[0]
    return ot.emd2(a, a, M, return_matrix=True, processes=10)[1]['G']


def compute_ot_matrix_par(x1, x2):
    arr = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(compute_ot_matrix)(x1[j], x2[j]) for j in range(x1.shape[0])
    )
    return np.concatenate([p[np.newaxis, ] for p in arr])


class PairwiseCost(nn.Module):

    def __init__(self, cost: Callable[[Tensor, Tensor], Tensor]):
        super().__init__()
        self.cost = cost

    def forward(self, x: Tensor, y: Tensor):

        B, N, D = x.shape

        assert y.shape[1] == N
        x = x[:, :, None, :]
        y = y[:, None, :, :]
        x = x.repeat(1, 1, N, 1).view(B, N * N, D)
        y = y.repeat(1, N, 1, 1).view(B, N * N, D)

        return self.cost(x, y).view(B, N, N)


class OTWasLoss(nn.Module):

    def forward(self, x1: Tensor, x2: Tensor) -> Loss:
        with torch.no_grad():
            P = compute_ot_matrix_par(x1.cpu().numpy(), x2.cpu().numpy())
            P = torch.from_numpy(P).type_as(x1).cuda()
        M = PairwiseCost(lambda t1, t2: (t1 - t2).pow(2).sum(dim=-1).sqrt())(x1, x2)
        return Loss((M * P).sum(dim=[1, 2]).mean())


class OTWasDist(nn.Module):

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        with torch.no_grad():
            P = compute_ot_matrix_par(x1.cpu().numpy(), x2.cpu().numpy())
            P = torch.from_numpy(P).type_as(x1).cuda()
        M = PairwiseCost(lambda t1, t2: (t1 - t2).pow(2).sum(dim=-1).sqrt())(x1, x2)
        return (M * P).sum(dim=[1, 2])