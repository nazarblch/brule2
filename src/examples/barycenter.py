
from torch.distributions import Dirichlet
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.replay_data import ReplayBuffer

from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch import optim
from dataset.lazy_loader import LazyLoader, MAFL, W300DatasetLoader
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2D01
from loss.weighted_was import WeightedSamplesLoss
from parameters.path import Paths

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

image_size = 256
batch_size = 64
padding = 68
MAFL.batch_size = batch_size
W300DatasetLoader.batch_size = batch_size

NC = 256

bc_net = nn.Sequential(
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

bc_net_opt = optim.Adam(bc_net.parameters(), lr=0.001)
sced = ReduceLROnPlateau(bc_net_opt)


replay_buf = ReplayBuffer(2)





# mafl_dataloader = LazyLoader.w300().loader_train_inf
# mes = UniformMeasure2D01(next(mafl_dataloader)['meta']['keypts_normalized'].type(torch.float32)).cuda()
mes = UniformMeasure2D01(next(iter(LazyLoader.celeba_test(batch_size)))[1]).cuda()


for j in range(10000):
    weights = Dirichlet(torch.ones(batch_size)/10).sample().cuda()
    barycenter, lll = compute_wbc(mes, weights, min(200, j + 10))

    if j % 50 == 0:
       print(j)
       sced.step(lll)

    if j % 50 == 0:
        plt.imshow(barycenter.toImage(200)[0][0].detach().cpu().numpy())
        plt.show()

    starting_model_number = 0
    if j % 1000 == 0 and j > 0:
        torch.save(
            bc_net.state_dict(),
            f'{Paths.default.models()}/bc_model256_64_{str(j + starting_model_number).zfill(6)}.pt',
        )