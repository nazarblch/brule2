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