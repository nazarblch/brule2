#%%
import argparse
import json
import math
import sys, os

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))

from argparse import ArgumentParser
from gan.nn.stylegan.components import ConvLayer, EqualLinear, ResBlock
import albumentations
from nn.common.view import View
from dataset.transforms import ToNumpy, NumpyBatch, ToTensor

from parameters.dataset import DatasetParameters
from parameters.model import ModelParameters
from parameters.run import RuntimeParameters
from models.autoencoder import StyleGanAutoEncoder
from loss.l1 import l1_loss
from loss.mes import MesBceWasLoss, MesBceL2Loss
from metrics.board import send_images_to_tensorboard, plot_img_with_lm

from loss.weighted_was import OTWasLoss
from metrics.landmarks import verka_300w, verka_300w_w2
from gan.loss.perceptual.psp import PSPLoss
from models.hg import HG_skeleton, HG_heatmap

from wr import WR

from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from torch import optim

import random
from gan.loss.loss_base import Loss
import numpy as np
import time
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton, heatmap_to_measure

from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, W300Landmarks
from dataset.probmeasure import UniformMeasure2D01
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel, CondStyleGanModel
from gan.nn.stylegan.discriminator import Discriminator, ConditionalDiscriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths


manualSeed = 79
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 16
image_size = 256
noise_size = 512


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        DatasetParameters(),
        RuntimeParameters(),
        ModelParameters()
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
W300Landmarks.batch_size = batch_size

starting_model_number = 90000
N = args.data_size
weights = torch.load(
    f'{Paths.default.models()}/lmgen_{N}_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

heatmapper = ToGaussHeatMap(256, 4)
hg = nn.Sequential(
    EqualLinear(100, 256, activation='fused_lrelu'),
    EqualLinear(256, 256, activation='fused_lrelu'),
    EqualLinear(256, 256, activation='fused_lrelu'),
    EqualLinear(256, 256, activation='fused_lrelu'),
    EqualLinear(256, 136),
    nn.Sigmoid(),
    View(68, 2)
)
hg.load_state_dict(weights['gh'])
hg = hg.cuda()
hm_discriminator = Discriminator(image_size, input_nc=1, channel_multiplier=1)
hm_discriminator.load_state_dict(weights["dh"])
hm_discriminator = hm_discriminator.cuda()

gan_model = StyleGanModel[nn.Module](hg, StyleGANLoss(hm_discriminator), (0.001, 0.0015))

writer = SummaryWriter(f"{Paths.default.board()}/lmgen{int(time.time())}")
WR.writer = writer

test_noise = torch.randn(batch_size, 100, device=device)

hm_accumulator = Accumulator(hg, decay=0.98, write_every=100)

#
# os.mkdir(f"{Paths.default.data()}/w300_gen_{N}")
# os.mkdir(f"{Paths.default.data()}/w300_gen_{N}/lmbc")
#
#
for i in range(7000//batch_size):
    noise = torch.randn(batch_size, 100, device=device)
    batch_ldmks = hg.forward(noise).detach().cpu().numpy()
    print(i)
    for j in range(batch_size):
        b: Tensor = batch_ldmks[j]
        # b[43] = b[42]
        print((b < 0.001).nonzero())
        np.save(f"{Paths.default.data()}/w300_gen_{N}/lmbc/{i*batch_size+j}.npy", b)


# for i in range(100000):
#
#     WR.counter.update(i)
#
#     landmarks = torch.clamp(next(LazyLoader.w300_landmarks(args.data_path).loader_train_inf).cuda(), max=1)
#
#     noise = torch.randn(batch_size, 100, device=device)
#     hm_pred = heatmapper.forward(hg.forward(noise)).sum(1, keepdim=True)
#     hm_ref = heatmapper.forward(landmarks).detach().sum(1, keepdim=True)
#
#     gan_model.discriminator_train([hm_ref], [hm_pred.detach()])
#     gan_model.generator_loss([hm_ref], [hm_pred])\
#         .minimize_step(gan_model.optimizer.opt_min)
#
#     hm_accumulator.step(i)
#
#     if i % 10000 == 0 and i > 0:
#         torch.save(
#             {
#                 'gh': hg.state_dict(),
#                 'dh': hm_discriminator.state_dict()
#             },
#             f'{Paths.default.models()}/lmgen_{args.data_size}_{str(i + starting_model_number).zfill(6)}.pt',
#         )
#
#     if i % 100 == 0:
#         print(i)
#         with torch.no_grad():
#
#             hm_pred_test = heatmapper.forward(hg.forward(test_noise)).sum(1, keepdim=True)
#             send_images_to_tensorboard(writer, hm_pred_test, "HM", i)
#             landmarks = torch.clamp(next(LazyLoader.w300_landmarks(args.data_path).loader_train_inf).cuda(), max=1)
#             hm_ref_test = heatmapper.forward(landmarks).detach().sum(1, keepdim=True)
#             send_images_to_tensorboard(writer, hm_ref_test, "HM Test", i)
#
#
