import time
from typing import Generic, TypeVar, Callable, Tuple, Dict, List
import torch
import torch.nn as nn
import numpy as np
from albumentations import DualTransform
# from conda.nn.dist import Dist
from geomloss import SamplesLoss
from torch import Tensor

from dataset.probmeasure import ProbabilityMeasure
from gan.loss.loss_base import Loss
from loss.weighted_was import compute_ot_matrix_par


class RegularizerObject:
    @staticmethod
    def __call__(func):
        return Regularizer(func)


class Regularizer(nn.Module):
    def __init__(self, func):
        super(Regularizer, self).__init__()
        self.forward = func

    def __add__(self, reg: nn.Module):
        def forward_add(*x):
            return self.forward(*x) + reg.forward(*x)
        return RegularizerObject.__call__(forward_add)

    def __mul__(self, v: float):
        def forward_mul(*x):
            return self.forward(*x) * v
        return RegularizerObject.__call__(forward_mul)

    def __matmul__(self, array: List[float]):
        def apply(index: int):
            return self.__mul__(array[min(index, len(array) - 1)])
        return Apply(apply)


class ApplyObject:
    @staticmethod
    def __call__(func):
        return Apply(func)


class Apply:
    def __init__(self, apply):
        self.apply = apply

    def __add__(self, other):
        if isinstance(other, Regularizer) or isinstance(other, RegularizerObject):
            return ApplyObject.__call__(lambda i: self.apply(i) + other)
        else:
            return ApplyObject.__call__(lambda i: self.apply(i) + other.apply(i))





class DualTransformRegularizer:

    @staticmethod
    def __call__(transform: DualTransform,
                 loss: Callable[[Dict[str, object], Tensor], Loss]):

        return RegularizerObject.__call__(lambda image, mask: loss(transform(image=image, mask=mask), image))


class UnoTransformRegularizer:

    @staticmethod
    def __call__(transform: DualTransform,
                 loss: Callable[[Dict[str, object], Tensor, Tensor], Loss]):

        return RegularizerObject.__call__(lambda image, latent: loss(transform(image=image), image, latent))


class StyleTransformRegularizer:

    @staticmethod
    def __call__(transform: DualTransform,
                 loss: Callable[[Dict[str, object], Tensor], Loss]):

        return RegularizerObject.__call__(lambda image: loss(transform(image=image), image))


class Samples_Loss(nn.Module):
    def __init__(self, blur=.01, scaling=.9, diameter=None, border=None, p: int = 2):
        super(Samples_Loss, self).__init__()
        self.border = border
        self.loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=False, diameter=diameter, p=p)

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):
        batch_loss = self.loss(m1.probability, m1.coord, m2.probability, m2.coord)
        if self.border:
            batch_loss = batch_loss[batch_loss > self.border]
            if batch_loss.shape[0] == 0:
                return Loss.ZERO()
        return Loss(batch_loss.mean())


class LinearTransformOT:

    @staticmethod
    def forward(pred: ProbabilityMeasure, targets: ProbabilityMeasure):

        with torch.no_grad():
            P = compute_ot_matrix_par(pred.centered().coord.cpu().numpy(), targets.centered().coord.cpu().numpy())
            P = torch.from_numpy(P).type_as(pred.coord).cuda()

        xs = pred.centered().coord
        xsT = xs.transpose(1, 2)
        xt = targets.centered().coord

        a: Tensor = pred.probability + 1e-8
        a /= a.sum(dim=1, keepdim=True)
        a = a.reshape(a.shape[0], -1, 1)

        A = torch.inverse(xsT.bmm(a * xs)).bmm(xsT.bmm(P.bmm(xt)))

        T = targets.mean() - pred.mean()

        return A.type_as(pred.coord), T.detach()


class BarycenterRegularizer:

    @staticmethod
    def __call__(barycenter, ct: float = 1, ca: float = 2, cw: float = 5):

        def loss(image: Tensor, mask: ProbabilityMeasure):

            with torch.no_grad():
                A, T = LinearTransformOT.forward(mask, barycenter)

            t_loss = Samples_Loss(scaling=0.8, border=0.0001)(mask, mask.detach() + T)
            a_loss = Samples_Loss(scaling=0.8, border=0.0001)(mask.centered(), mask.centered().multiply(A).detach())
            w_loss = Samples_Loss(scaling=0.85, border=0.00001)(mask.centered().multiply(A), barycenter.centered().detach())

            # print(time.time() - t1)

            return a_loss * ca + w_loss * cw + t_loss * ct

        return RegularizerObject.__call__(loss)