import torch.nn as nn
import torch
from dataset.probmeasure import ProbabilityMeasure
from geomloss import SamplesLoss

from gan.loss.loss_base import Loss


class WeightedSamplesLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, blur=.01, scaling=.9, diameter=None, p: int = 2):
        super(WeightedSamplesLoss, self).__init__()
        self.weights = weights
        self.loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=False, diameter=diameter, p=p)

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):
        batch_loss = self.loss(m1.probability, m1.coord, m2.probability, m2.coord)
        return Loss((batch_loss * self.weights).sum())