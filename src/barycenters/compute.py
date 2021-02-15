from typing import List, Tuple
from torch import Tensor, nn
from torch import optim
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2D01
from loss.weighted_was import WeightedSamplesLoss


class Barycenters:
    NC = 256

    def __init__(self, batch_size, padding=68):

        NC = Barycenters.NC
        self.padding = padding
        self.batch_size = batch_size

        self.bc_net = nn.Sequential(
            nn.Linear(batch_size, NC),
            nn.ReLU(inplace=True),
            nn.Linear(NC, NC),
            nn.ReLU(inplace=True),
            nn.Linear(NC, NC),
            nn.ReLU(inplace=True),
            nn.Linear(NC, NC),
            nn.ReLU(inplace=True),
            nn.Linear(NC, padding * 2),
            nn.Sigmoid()
        ).cuda()

        self.bc_net_opt = optim.Adam(self.bc_net.parameters(), lr=0.001)
        # sced = ReduceLROnPlateau(bc_net_opt)

    def compute_wbc(self, measures: ProbabilityMeasure, weights: Tensor, opt_iters: int, initial: Tensor = None) -> ProbabilityMeasure:

        fabric = ProbabilityMeasureFabric(256)
        barycenter: ProbabilityMeasure = fabric.random(self.padding).cuda()
        if initial is not None:
            barycenter.coord = initial

        barycenter.requires_grad_()

        coord = barycenter.coord
        opt = optim.Adam(iter([coord]), lr=0.0005)

        for _ in range(opt_iters):

            barycenter_cat = fabric.cat([barycenter] * self.batch_size)

            WeightedSamplesLoss(weights)(barycenter_cat, measures).minimise_step(opt)

            barycenter.probability.data = barycenter.probability.relu().data
            barycenter.probability.data /= barycenter.probability.sum(dim=1, keepdim=True)

        return barycenter