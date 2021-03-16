from functools import reduce

import ot
import numpy as np
from typing import List

import torch

from dataset.toheatmap import ToGaussHeatMap
from models.autoencoder import StyleGanAutoEncoder
from parameters.path import Paths


class Uniform2DBarycenterSampler:

    def __init__(self, size: int, dir_alpha: float = 1):

        self.X = np.random.uniform(size=(size, 2))
        self.prob = np.ones(size) / size
        self.dir_alpha = dir_alpha

    def sample(self, measures: List[np.ndarray]) -> (np.ndarray, np.ndarray):

        weights = np.random.dirichlet([self.dir_alpha] * len(measures))

        return ot.lp.free_support_barycenter(
            measures,
            [self.prob] * len(measures),
            self.X,
            self.prob,
            weights=weights,
            numItermax=300
        ), weights

    def mean(self, measures: List[np.ndarray]) -> np.ndarray:

        return ot.lp.free_support_barycenter(
            measures,
            [self.prob] * len(measures),
            self.X,
            self.prob
        )


class Uniform2DAverageSampler:

    def __init__(self, size: int, dir_alpha: float = 1):

        self.X = np.random.uniform(size=(size, 2))
        self.dir_alpha = dir_alpha

    def sample(self, measures: List[np.ndarray], weights=None) -> (np.ndarray, np.ndarray):

        if weights is None:
            weights = np.random.dirichlet([self.dir_alpha] * len(measures))

        return reduce(lambda x, y: x+y, [measures[i] * weights[i] for i in range(len(measures))]), weights


class ImageBarycenterSampler:

    def __init__(self, lm_count, dir_alpha = 1):
        starting_model_number = 560000 + 50000 + 90000 + 30000
        weights = torch.load(
            f'{Paths.default.models()}/hm2img_{str(starting_model_number).zfill(6)}.pt',
            map_location="cpu"
        )

        self.enc_dec = StyleGanAutoEncoder(weights, load_style=True)

        self.dir_alpha = dir_alpha
        self.mes_sampler = Uniform2DBarycenterSampler(lm_count, dir_alpha)
        self.heatmapper = ToGaussHeatMap(256, 4)

    def sample(self, measures: List[np.ndarray], images: torch.Tensor) -> (np.ndarray, torch.Tensor, np.ndarray):
        bc_mes, weights = self.mes_sampler.sample(measures)
        with torch.no_grad():
            latents = self.enc_dec.encode_latent(images)
            bc_latent = reduce(lambda x, y: x + y,
                   [(latents[i] * weights[i]).type(torch.float32) for i in range(len(measures))]
                   )[None,]
            bc_mes_cuda = torch.from_numpy(bc_mes).cuda().type(torch.float32)[None, ]
            hm = self.heatmapper.forward(bc_mes_cuda).sum(1, keepdim=True)

            return bc_mes, self.enc_dec.decode(hm, bc_latent)[0].cpu(), weights