import torch
from torch import nn

from dataset.cfinder import ContFinder
from dataset.lazy_loader import BraTSLoader
from loss.weighted_was import OTWasDist


class DiceSum(nn.Module):
    def __init__(self, threshold=0.5):
        super(DiceSum, self).__init__()
        self.threshold = threshold

    def forward(self, prediction, target):
        smooth = torch.tensor(1e-15).float()
        target = (target > 1e-10).float()
        prediction = (prediction > self.threshold).float()
        dice_part = (2*torch.sum(prediction * target, dim=(1, 2, 3)) + smooth) / \
                    (torch.sum(prediction, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + smooth)
        return dice_part


def verka_segm(enc, loader: BraTSLoader):
    sum_loss = 0
    n = len(loader.dataset_test)
    for i, batch in enumerate(loader.test_loader):
        data = batch[0].cuda()
        segm_ref = batch[1].cuda()
        pred = enc(data)
        sum_loss += DiceSum().forward(pred, segm_ref).sum().item()
    print("test loss: ", sum_loss / n)
    return sum_loss / n


def verka_segm_cont(enc, loader: BraTSLoader):
    sum_loss = 0
    n = len(loader.dataset_test)
    for i, batch in enumerate(loader.test_loader):
        data = batch[0].cuda()
        segm_ref = batch[1].cuda()
        landmarks = ContFinder.get_conturs_batch(segm_ref).coord
        pred = enc(data)["mes"].coord
        sum_loss += OTWasDist().forward(pred, landmarks).sum().item()
    print("test loss: ", sum_loss / n)
    return sum_loss / n