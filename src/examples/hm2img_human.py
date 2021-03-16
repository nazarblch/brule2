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
from metrics.landmarks import verka_300w, verka_300w_w2, verka_human
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
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, W300Landmarks, HumanLoader
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
image_size = 128
measure_size = 32
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
args.image_size = image_size
args.batch_size = batch_size
args.measure_size = measure_size

for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
HumanLoader.batch_size = batch_size


g_transforms = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.3, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=15),
        ])),
        ToTensor(device),
    ])

starting_model_number = 90000 + args.weights
weights = torch.load(
    f'{Paths.default.models()}/human_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

enc_dec = StyleGanAutoEncoder(hm_nc=measure_size, image_size=image_size).load_state_dict(weights).cuda()

discriminator_img = Discriminator(image_size)
discriminator_img.load_state_dict(weights['di'])
discriminator_img = discriminator_img.cuda()

heatmapper = ToGaussHeatMap(image_size, 2)
hg = HG_heatmap(heatmapper, num_classes=measure_size, image_size=image_size)
hg.load_state_dict(weights['gh'])
hg = hg.cuda()
hm_discriminator = Discriminator(image_size, input_nc=measure_size, channel_multiplier=1)
hm_discriminator.load_state_dict(weights["dh"])
hm_discriminator = hm_discriminator.cuda()

gan_model_tuda = StyleGanModel[HeatmapToImage](enc_dec.generator, StyleGANLoss(discriminator_img), (0.002, 0.003))
gan_model_obratno = StyleGanModel[HG_skeleton](hg, StyleGANLoss(hm_discriminator), (4e-5, 0.0015/4))

style_opt = optim.Adam(enc_dec.style_encoder.parameters(), lr=1e-5)

writer = SummaryWriter(f"{Paths.default.board()}/human{int(time.time())}")
WR.writer = writer

batch = next(LazyLoader.human36().loader_train_inf)
test_img = batch["A"].cuda()
test_landmarks = torch.clamp(batch["paired_B"].cuda(), max=1)
test_hm = heatmapper.forward(test_landmarks).detach()
test_noise = mixing_noise(batch_size, 512, 0.9, device)

psp_loss = PSPLoss(id_lambda=0, lpips_lambda=0).cuda()
mes_loss = MesBceL2Loss(heatmapper, bce_coef=1000000/2, l2_coef=2000)

image_accumulator = Accumulator(enc_dec.generator, decay=0.9, write_every=100)
hm_accumulator = Accumulator(hg, decay=0.95, write_every=100)


for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.human36().loader_train_inf)
    real_img = batch["A"].cuda()
    cond_img = batch["cond_A"].cuda()
    landmarks = torch.clamp(batch["paired_B"].cuda(), max=1)
    heatmap = heatmapper.forward(landmarks).detach()

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/cycle_loss.json")))

    fake, fake_latent = enc_dec.generate(heatmap)
    fake_latent_pred = enc_dec.encode_latent(fake)

    gan_model_tuda.discriminator_train([real_img], [fake.detach()])
    (
        gan_model_tuda.generator_loss([real_img], [fake]) +
        l1_loss(fake_latent_pred, fake_latent) * coefs["style"]
    ).minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    hm_pred = hg.forward(real_img)["hm"]
    hm_ref = heatmapper.forward(landmarks).detach()
    gan_model_obratno.discriminator_train([hm_ref], [hm_pred.detach()])
    gan_model_obratno.generator_loss([hm_ref], [hm_pred]).__mul__(coefs["obratno"])\
        .minimize_step(gan_model_obratno.optimizer.opt_min)

    fake2, _ = enc_dec.generate(heatmap)
    WR.writable("cycle", mes_loss.forward)(hg.forward(fake2)["mes"], UniformMeasure2D01(landmarks)).__mul__(coefs["hm"])\
        .minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    latent = enc_dec.encode_latent(cond_img)
    restored = enc_dec.decode(hg.forward(real_img)["hm"], latent)
    WR.writable("cycle2", psp_loss.forward)(real_img, real_img, restored, latent).__mul__(coefs["img"])\
        .minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min, style_opt)

    image_accumulator.step(i)
    hm_accumulator.step(i)

    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'gi': enc_dec.generator.state_dict(),
                'gh': hg.state_dict(),
                'di': discriminator_img.state_dict(),
                'dh': hm_discriminator.state_dict(),
                's': enc_dec.style_encoder.state_dict()
            },
            f'{Paths.default.models()}/human_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl2 = verka_human(hg)
            writer.add_scalar("verka", tl2, i)

            sk_pred = hg.forward(test_img)["hm"]
            fake, _ = enc_dec.generate(test_hm, test_noise)
            latent = enc_dec.encode_latent(test_img)
            restored = enc_dec.decode(sk_pred, latent)

            send_images_to_tensorboard(writer, fake + test_hm.sum(1, keepdim=True), "FAKE", i)
            send_images_to_tensorboard(writer, test_img + sk_pred.sum(1, keepdim=True), "REAL", i)
            send_images_to_tensorboard(writer, restored + sk_pred.sum(1, keepdim=True), "RESTORED", i)

