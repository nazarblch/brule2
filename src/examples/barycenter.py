
from torch.distributions import Dirichlet
from torch.optim.lr_scheduler import ReduceLROnPlateau

from barycenters.compute import Barycenters
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
W300DatasetLoader.batch_size = batch_size

# Download 300w from https://ibug.doc.ic.ac.uk/resources/300-W/
# Set path in {Paths.default.data()}/300w
batch = next(LazyLoader.w300().loader_train_inf)
landmarks = batch["meta"]["keypts_normalized"].cuda()
measure = UniformMeasure2D01(torch.clamp(landmarks, max=1))

bc_tool = Barycenters(batch_size, padding)

weights = Dirichlet(torch.ones(batch_size) / 10).sample().cuda()
barycenter = bc_tool.compute_wbc(measure, weights, 200)
plt.imshow(barycenter.toImage(200)[0][0].detach().cpu().numpy())
plt.show()




