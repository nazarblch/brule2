#%%
import argparse
import json
import math
import sys, os

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))


from gan.models.base import requires_grad
from loss.reg import DualTransformRegularizer, BarycenterRegularizer

from argparse import ArgumentParser

import albumentations

from dataset.transforms import ToNumpy, NumpyBatch, ToTensor

from parameters.dataset import DatasetParameters
from parameters.model import ModelParameters
from parameters.run import RuntimeParameters
from models.autoencoder import StyleGanAutoEncoder
from loss.l1 import l1_loss
from loss.mes import MesBceWasLoss, MesBceL2Loss, coord_hm_loss
from metrics.board import send_images_to_tensorboard, plot_img_with_lm

from loss.weighted_was import OTWasLoss
from metrics.landmarks import verka_300w, verka_300w_w2, verka_300w_w2_boot, verka_cardio_w2
from gan.loss.perceptual.psp import PSPLoss
from models.hg import HG_skeleton, HG_heatmap

from wr import WR

from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from torch import optim

import random
from gan.loss.loss_base import Loss
import numpy as np
import time
from dataset.toheatmap import ToGaussHeatMap,  heatmap_to_measure

from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, W300Landmarks, Cardio
from dataset.probmeasure import UniformMeasure2D01, UniformMeasure2DFactory
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

batch_size = 8
image_size = 256
noise_size = 512
n_mlp = 8
lr_mlp = 0.01


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


starting_model_number = 264000
weights = torch.load(
    f'{Paths.default.models()}/cardio_brule_sup_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

heatmapper = ToGaussHeatMap(256, 4)
hg = HG_heatmap(heatmapper, num_blocks=1, num_classes=200)
hg.load_state_dict(weights['gh'])
hg = hg.cuda()
cont_opt = optim.Adam(hg.parameters(), lr=2e-5, betas=(0, 0.8))

# gan_model_tuda = StyleGanModel[HeatmapToImage](enc_dec.generator, StyleGANLoss(discriminator_img), (0.001/4, 0.0015/4))

# style_opt = optim.Adam(enc_dec.style_encoder.parameters(), lr=1e-5)

writer = SummaryWriter(f"{Paths.default.board()}/brule1_cardio_{int(time.time())}")
WR.writer = writer

batch = next(iter(LazyLoader.cardio().test_loader))
test_img = batch["image"].cuda()


# psp_loss = PSPLoss(id_lambda=0).cuda()
mes_loss = MesBceWasLoss(heatmapper, bce_coef=10000, was_coef=100)

# image_accumulator = Accumulator(enc_dec.generator, decay=0.99, write_every=100)
hm_accumulator = Accumulator(hg, decay=0.99, write_every=100)


for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.cardio().loader_train_inf)
    real_img = batch["image"].cuda()
    train_landmarks = batch["keypoints"].cuda()

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/cycle_loss_2.json")))

    WR.writable("sup", mes_loss.forward)(hg.forward(real_img)["mes"], UniformMeasure2D01(train_landmarks)).__mul__(coefs["sup"]) \
        .minimize_step(cont_opt)

    hm_accumulator.step(i)

    if i % 1000 == 0 and i > 0:
        torch.save(
            {
                'gh': hg.state_dict(),
            },
            f'{Paths.default.models()}/cardio_brule_sup_{str(i + starting_model_number).zfill(6)}.pt',
        )


    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl2 = verka_cardio_w2(hg)
            writer.add_scalar("verka", tl2, i)

            heatmapper22 = ToGaussHeatMap(256, 2)
            sk_pred = heatmapper22.forward(hg.forward(test_img)["mes"].coord).sum(1, keepdim=True)

            send_images_to_tensorboard(writer, test_img + sk_pred, "REAL", i)




