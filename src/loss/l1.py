from gan.loss.loss_base import Loss
from torch import nn


def l1_loss(pred, target):
    return Loss(nn.L1Loss().forward(pred, target))