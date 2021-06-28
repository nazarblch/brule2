from dataset.probmeasure import UniformMeasure2DFactory
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton
from gan.loss.loss_base import Loss
from loss.weighted_was import OTWasLoss
from torch import nn, Tensor


class MesBceL2Loss(nn.Module):

    def __init__(self, heatmapper, bce_coef: float = 1000000, l2_coef: float = 2000):
        super().__init__()
        self.heatmapper = heatmapper
        self.bce_coef = bce_coef
        self.l2_coef = l2_coef

    def forward(self, pred_mes, target_mes) -> Loss:
        pred_hm = self.heatmapper.forward(pred_mes.coord)
        target_hm = self.heatmapper.forward(target_mes.coord)

        pred_hm: Tensor = pred_hm / (pred_hm.sum(dim=[1, 2, 3], keepdim=True).detach() + 1e-8)
        target_hm: Tensor = target_hm / target_hm.sum(dim=[1, 2, 3], keepdim=True).detach()

        return Loss(
            nn.BCELoss()(pred_hm, target_hm) * self.bce_coef * pred_hm.shape[1] +
            (pred_mes.coord - target_mes.coord).pow(2).sum(dim=2).sqrt().mean() * self.l2_coef
            # nn.MSELoss()(pred_mes.coord, target_mes.coord) * self.l2_coef
        )


class MesBceWasLoss(nn.Module):

    def __init__(self, heatmapper, bce_coef: float = 1000000, was_coef: float = 2000):
        super().__init__()
        self.heatmapper = heatmapper
        self.bce_coef = bce_coef
        self.was_coef = was_coef

    def forward(self, pred_mes, target_mes) -> Loss:
        pred_hm = self.heatmapper.forward(pred_mes.coord).sum(dim=[1], keepdim=True)
        target_hm = self.heatmapper.forward(target_mes.coord).sum(dim=[1], keepdim=True)

        pred_hm = pred_hm / (pred_hm.sum(dim=[1, 2, 3], keepdim=True).detach() + 1e-8)
        target_hm = target_hm / target_hm.sum(dim=[1, 2, 3], keepdim=True).detach()

        return Loss(
            nn.BCELoss()(pred_hm, target_hm) * self.bce_coef +
            OTWasLoss()(pred_mes.coord, target_mes.coord).to_tensor() * self.was_coef
        )


def noviy_hm_loss(pred, target, coef=1.0):

    pred = pred / pred.sum(dim=[2, 3], keepdim=True).detach()
    target = target / target.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred, target) * coef
    )


def coord_hm_loss(pred_coord: Tensor, target_hm: Tensor, coef=1.0):
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True)
    target_hm = target_hm.detach()

    heatmapper = ToGaussHeatMap(256, 4)

    target_coord = UniformMeasure2DFactory.from_heatmap(target_hm).coord.detach()
    # sk = CoordToGaussSkeleton(target_hm.shape[-1], 1)
    # pred_sk = sk.forward(pred_coord).sum(dim=1, keepdim=True)
    # target_sk = sk.forward(target_coord).sum(dim=1, keepdim=True).detach()
    pred_hm = heatmapper.forward(pred_coord).sum(dim=1, keepdim=True)
    pred_hm = pred_hm / pred_hm.sum(dim=[2, 3], keepdim=True).detach()
    target_hm = heatmapper.forward(target_coord).sum(dim=1, keepdim=True).detach()
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred_hm, target_hm) * coef * 1.5 +
        # noviy_hm_loss(pred_sk, target_sk, coef).to_tensor() * 0.5 +
        nn.MSELoss()(pred_coord, target_coord) * (0.001 * coef) +
        nn.L1Loss()(pred_coord, target_coord) * (0.001 * coef)
    )