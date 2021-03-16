#%%
import argparse
import json
import math
import sys, os

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))

from argparse import ArgumentParser

import albumentations

from dataset.transforms import ToNumpy, NumpyBatch, ToTensor

from parameters.dataset import DatasetParameters
from parameters.model import ModelParameters
from parameters.run import RuntimeParameters
from models.autoencoder import StyleGanAutoEncoder
from loss.l1 import l1_loss
from loss.mes import MesBceWasLoss, MesBceL2Loss
from metrics.board import send_images_to_tensorboard

from loss.weighted_was import OTWasLoss
from metrics.landmarks import verka_300w, verka_300w_w2, verka_cardio_w2
from gan.loss.perceptual.psp import PSPLoss
from train_procedure import gan_trainer, content_trainer_with_gan, coord_hm_loss
from models.hg import HG_skeleton, HG_heatmap

from wr import WR

from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from torch import optim

import random
from gan.loss.loss_base import Loss
import numpy as np
import time
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton

from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, W300Landmarks, Cardio
from dataset.probmeasure import UniformMeasure2D01
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel, CondStyleGanModel
from gan.nn.stylegan.discriminator import Discriminator, ConditionalDiscriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths


manualSeed = 72
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 8
image_size = 256

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
Cardio.batch_size = batch_size


heatmapper = ToGaussHeatMap(256, 4)
hg = HG_heatmap(heatmapper, num_classes=200)
hg = hg.cuda()

hg_opt = optim.Adam(hg.parameters(), lr=1e-4)

writer = SummaryWriter(f"{Paths.default.board()}/hm2img{int(time.time())}")
WR.writer = writer

batch = next(LazyLoader.cardio().loader_train_inf)
test_img = batch["image"].cuda()
test_landmarks = torch.clamp(batch["keypoints"].cuda(), max=1)
test_hm = heatmapper.forward(test_landmarks).sum(1, keepdim=True).detach()
test_noise = mixing_noise(batch_size, 512, 0.9, device)

mes_loss = MesBceWasLoss(heatmapper, bce_coef=100000, was_coef=200)

for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.cardio().loader_train_inf)
    real_img = batch["image"].cuda()
    landmarks = torch.clamp(batch["keypoints"].cuda(), max=1)

    WR.writable("cycle", mes_loss.forward)(hg.forward(real_img)["mes"], UniformMeasure2D01(landmarks))\
        .minimize_step(hg_opt)

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl2 = verka_cardio_w2(hg)
            writer.add_scalar("verka", tl2, i)

            sk_pred = hg.forward(test_img)["hm_sum"]
            send_images_to_tensorboard(writer, test_img + sk_pred, "REAL", i)


