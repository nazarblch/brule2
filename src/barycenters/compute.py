from typing import List, Tuple
from torch import Tensor, nn
from torch import optim
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2D01
from dataset.replay_data import ReplayBuffer
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
        self.replay_buf = ReplayBuffer(2)
        # sced = ReduceLROnPlateau(bc_net_opt)

    def train(self, measures: ProbabilityMeasure, weights: Tensor, opt_iters: int):
        pred = self.bc_net(weights[None, :]).reshape(1, self.padding, 2).detach()
        ref = self.compute_wbc(measures, weights, opt_iters, pred)

        self.replay_buf.append(weights.cpu().detach()[None, :], ref.coord.cpu().detach())

        lll = None

        if self.replay_buf.size() > 32:
            ws, bs = self.replay_buf.sample(32)

            self.bc_net.zero_grad()
            ll = (self.bc_net(ws).reshape(-1, self.padding, 2) - bs).pow(2).sum() / 32
            lll = ll.item()
            ll.backward()
            self.bc_net_opt.step()

        return ref, lll

    def compute_wbc(self, measures: ProbabilityMeasure, weights: Tensor, opt_iters: int, initial: Tensor = None) -> ProbabilityMeasure:

        fabric = ProbabilityMeasureFabric(256)
        barycenter: ProbabilityMeasure = fabric.random(self.padding).cuda()
        if initial is not None:
            barycenter.coord = initial

        barycenter.requires_grad_()

        coord = barycenter.coord
        opt = optim.Adam(iter([coord]), lr=0.005)

        for _ in range(opt_iters):

            barycenter_cat = fabric.cat([barycenter] * self.batch_size)

            WeightedSamplesLoss(weights)(barycenter_cat, measures).minimize_step(opt)

            barycenter.probability.data = barycenter.probability.relu().data
            barycenter.probability.data /= barycenter.probability.sum(dim=1, keepdim=True)

        return barycenter