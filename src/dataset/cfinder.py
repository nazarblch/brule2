import numpy as np
import cv2 as cv
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from dataset.probmeasure import UniformMeasure2D01
from dataset.toheatmap import ToGaussHeatMap


def concat(contours):
    return np.concatenate(contours)


def take_longest(contours):
    max_cont = contours[0]
    for c in contours:
        if c.shape[0] > max_cont.shape[0]:
                max_cont = c

    return max_cont


class ContFinder:

    N = 200
    MODE = cv.RETR_EXTERNAL
    POSTPROC = concat
    MASK_THRASH = 0.5

    @staticmethod
    def get_contours(mask: np.ndarray):

        mask[mask > ContFinder.MASK_THRASH] = 1
        mask[mask < ContFinder.MASK_THRASH] = 0
        mask = mask[0].astype(np.uint8)

        contours, hierarchy = cv.findContours(mask, 0, cv.CHAIN_APPROX_NONE)

        max_cont = ContFinder.POSTPROC(contours)

        n = max_cont.shape[0]
        idx = np.linspace(n - 1, 0, ContFinder.N).astype(int)
        res = max_cont[idx, 0, :].astype(np.float32)

        assert res.shape[0] == ContFinder.N
        return res

    @staticmethod
    def get_conturs_batch(mask: Tensor):

        B = mask.shape[0]
        mask = mask.cpu().numpy()

        coord = torch.cat([torch.from_numpy(ContFinder.get_contours(mask[i]))[None, ] for i in range(B)]).cuda()
        return UniformMeasure2D01(coord / (mask.shape[-1] - 1))


def fill_contour(landmarks: np.ndarray):

    heatmapper = ToGaussHeatMap(256, 3)
    hm = heatmapper.forward(torch.from_numpy(landmarks)[None, ])[0].numpy()

    img_pl = np.zeros((256, 256))
    hmi = hm.sum(0)
    res = cv.findContours(hmi.astype(np.uint8), 0, cv.CHAIN_APPROX_NONE)
    contours = res[-2]
    contours = [i.transpose(1, 0, 2) for i in contours]
    cv.fillPoly(img_pl, contours, 1)

    return img_pl





