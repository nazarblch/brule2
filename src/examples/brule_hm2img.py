#%%
import argparse
import json
import math
import sys, os


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))

from gan.models.base import requires_grad
# from unsupervised_segmentation.brule_loss.regulariser import DualTransformRegularizer, BarycenterRegularizer
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
from metrics.landmarks import verka_300w, verka_300w_w2, verka_300w_w2_boot
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
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, W300Landmarks
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
W300DatasetLoader.batch_size = batch_size
W300Landmarks.batch_size = batch_size

g_transforms = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.3, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=15),
        ])),
        ToTensor(device),
    ])



starting_model_number = 560000 + args.weights
weights = torch.load(
    f'{Paths.default.models()}/hm2img_brule1_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

enc_dec = StyleGanAutoEncoder().load_state_dict(weights).cuda()

discriminator_img = Discriminator(image_size)
discriminator_img.load_state_dict(weights['di'])
discriminator_img = discriminator_img.cuda()

heatmapper = ToGaussHeatMap(256, 4)
hg = HG_heatmap(heatmapper, num_blocks=1)
hg.load_state_dict(weights['gh'])
hg = hg.cuda()
cont_opt = optim.Adam(hg.parameters(), lr=2e-5, betas=(0, 0.8))

gan_model_tuda = StyleGanModel[HeatmapToImage](enc_dec.generator, StyleGANLoss(discriminator_img), (0.001/4, 0.0015/4))

style_opt = optim.Adam(enc_dec.style_encoder.parameters(), lr=1e-5)

writer = SummaryWriter(f"{Paths.default.board()}/brule1_{int(time.time())}")
WR.writer = writer

batch = next(LazyLoader.w300().loader_train_inf)
test_img = batch["data"].cuda()
test_landmarks = torch.clamp(next(LazyLoader.w300_landmarks(args.data_path).loader_train_inf).cuda(), max=1)
test_hm = heatmapper.forward(test_landmarks).sum(1, keepdim=True).detach()
test_noise = mixing_noise(batch_size, 512, 0.9, device)

psp_loss = PSPLoss().cuda()
mes_loss = MesBceWasLoss(heatmapper, bce_coef=10000, was_coef=100)

image_accumulator = Accumulator(enc_dec.generator, decay=0.99, write_every=100)
hm_accumulator = Accumulator(hg, decay=0.99, write_every=100)


g_transforms_2: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.7, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.9, rotate_limit=15),
        ])),
        ToTensor(device),
    ])

R_t = DualTransformRegularizer.__call__(
    g_transforms_2, lambda trans_dict, img:
    coord_hm_loss(hg(trans_dict['image'])["mes"].coord, trans_dict['mask'])
)


barycenter: UniformMeasure2D01 = UniformMeasure2DFactory.load(
    f"{Paths.default.models()}/face_barycenter_68").cuda().batch_repeat(batch_size)

R_b = BarycenterRegularizer.__call__(barycenter, 1.0, 2.0, 4.0)


def train_content(cont_opt, R_b, R_t, real_img, encoder_HG):

    # heatmapper = ToGaussHeatMap(256, 4)
    requires_grad(encoder_HG, True)

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/content_loss.json")))
    encoded = encoder_HG(real_img)
    pred_measures: UniformMeasure2D01 = encoded["mes"]

    heatmap_content = encoded["hm"]

    ll = (
        WR.writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"] +
        WR.writable("R_t", R_t.__call__)(real_img, heatmap_content) * coefs["R_t"]
    )

    ll.minimize_step(cont_opt)


for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.w300().loader_train_inf)
    real_img = batch["data"].cuda()
    # landmarks = torch.clamp(next(LazyLoader.w300_landmarks(args.data_path).loader_train_inf).cuda(), max=1)
    # heatmap_sum = heatmapper.forward(landmarks).sum(1, keepdim=True).detach()

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/cycle_loss_2.json")))

    with torch.no_grad():
        pred = hg.forward(real_img)
        hm_pred = pred["hm_sum"].detach()
        mes_pred = pred["mes"].detach()

    fake, fake_latent = enc_dec.generate(hm_pred)
    fake_latent_pred = enc_dec.encode_latent(fake)

    gan_model_tuda.discriminator_train([real_img], [fake.detach()])
    (
        gan_model_tuda.generator_loss([real_img], [fake]) +
        l1_loss(fake_latent_pred, fake_latent) * coefs["style"]
    ).minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    train_content(cont_opt, R_b, R_t, real_img, hg)

    fake2, _ = enc_dec.generate(hm_pred)
    WR.writable("cycle", mes_loss.forward)(hg.forward(fake2)["mes"], mes_pred).__mul__(coefs["hm"]) \
        .minimize_step(gan_model_tuda.optimizer.opt_min, cont_opt)

    latent = enc_dec.encode_latent(g_transforms(image=real_img)["image"])
    restored = enc_dec.decode(hg.forward(real_img)["hm_sum"], latent)
    WR.writable("cycle2", psp_loss.forward)(real_img, real_img, restored, latent).__mul__(coefs["img"])\
        .minimize_step(gan_model_tuda.optimizer.opt_min, cont_opt, style_opt)

    image_accumulator.step(i)
    hm_accumulator.step(i)

    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'gi': enc_dec.generator.state_dict(),
                'gh': hg.state_dict(),
                'di': discriminator_img.state_dict(),
                # 'dh': hm_discriminator.state_dict(),
                's': enc_dec.style_encoder.state_dict()
            },
            f'{Paths.default.models()}/hm2img_brule1_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl2 = verka_300w_w2(hg)
            writer.add_scalar("verka", tl2, i)

            sk_pred = hg.forward(test_img)["hm_sum"]
            fake, _ = enc_dec.generate(test_hm, test_noise)
            latent = enc_dec.encode_latent(test_img)
            restored = enc_dec.decode(sk_pred, latent)

            send_images_to_tensorboard(writer, fake + test_hm, "FAKE", i)
            send_images_to_tensorboard(writer, test_img + sk_pred, "REAL", i)
            send_images_to_tensorboard(writer, restored + sk_pred, "RESTORED", i)

