#%%
import argparse
import json
import math
import sys, os

import albumentations

sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/gans/'))

from parameters.dataset import DatasetParameters
from parameters.model import ModelParameters
from parameters.run import RuntimeParameters
from dataset.transforms import ToNumpy, NumpyBatch, ToTensor

from loss.mes import MesBceWasLoss
from metrics.board import send_images_to_tensorboard, plot_img_with_lm
from metrics.landmarks import verka_cardio_w2
from models.autoencoder import StyleGanAutoEncoder

from loss.weighted_was import OTWasLoss, compute_ot_matrix_par, PairwiseCost
from gan.loss.perceptual.psp import PSPLoss
from train_procedure import gan_trainer, content_trainer_with_gan, coord_hm_loss
from models.hg import HG_skeleton, HG_heatmap
from wr import WR
from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from torch import optim
import random
from gan.loss.loss_base import Loss

import time
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton
import numpy as np
from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, Cardio, CardioLandmarks
from dataset.probmeasure import UniformMeasure2D01
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel, CondStyleGanModel
from gan.nn.stylegan.discriminator import Discriminator, ConditionalDiscriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths


def l1_loss(pred, target):
    return Loss(nn.L1Loss().forward(pred, target))

def l2_loss(pred, target):
    return Loss(nn.MSELoss().forward(pred, target))


manualSeed = 71
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 4
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
CardioLandmarks.batch_size = batch_size

g_transforms = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.3, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=15),
        ])),
        ToTensor(device),
    ])

starting_model_number = 160000 + 90000
weights = torch.load(
    f'{Paths.default.models()}/cardio_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

enc_dec = StyleGanAutoEncoder().load_state_dict(weights).cuda()

discriminator_img = Discriminator(image_size)
discriminator_img.load_state_dict(weights['di'])
discriminator_img = discriminator_img.cuda()

heatmapper = ToGaussHeatMap(256, 4)
hg = HG_heatmap(heatmapper, num_classes=200)
hg.load_state_dict(weights['gh'])
hg = hg.cuda()
hm_discriminator = Discriminator(image_size, input_nc=1, channel_multiplier=1)
hm_discriminator.load_state_dict(weights["dh"])
hm_discriminator = hm_discriminator.cuda()

gan_model_tuda = StyleGanModel[HeatmapToImage](enc_dec.generator, StyleGANLoss(discriminator_img), (0.001, 0.0015))
gan_model_obratno = StyleGanModel[HG_skeleton](hg, StyleGANLoss(hm_discriminator), (2e-5, 0.0015/4))

style_opt = optim.Adam(enc_dec.style_encoder.parameters(), lr=1e-5)

writer = SummaryWriter(f"{Paths.default.board()}/cardio{int(time.time())}")
WR.writer = writer

#%%

test_batch = next(LazyLoader.cardio().loader_train_inf)
test_img = test_batch["image"].cuda()
test_landmarks = next(LazyLoader.cardio_landmarks().loader_train_inf).cuda()
test_measure = UniformMeasure2D01(torch.clamp(test_landmarks, max=1))
test_hm = heatmapper.forward(test_measure.coord).sum(1, keepdim=True).detach()
test_noise = mixing_noise(batch_size, 512, 0.9, device)

psp_loss = PSPLoss(id_lambda=0).cuda()
mes_loss = MesBceWasLoss(heatmapper, bce_coef=1000000, was_coef=2000)

image_accumulator = Accumulator(enc_dec.generator, decay=0.99, write_every=100)
hm_accumulator = Accumulator(hg, decay=0.99, write_every=100)

fake, _ = enc_dec.generate(test_hm, test_noise)
plt_img = torch.cat([test_img[:3], fake[:3]]).detach().cpu()
plt_lm = torch.cat([hg.forward(test_img)["mes"].coord[:3], test_landmarks[:3]]).detach().cpu()
plot_img_with_lm(plt_img, plt_lm, nrows=2, ncols=3)

for i in range(100000):

    WR.counter.update(i)

    real_img = next(LazyLoader.cardio().loader_train_inf)["image"].cuda()
    landmarks = next(LazyLoader.cardio_landmarks().loader_train_inf).cuda()
    measure = UniformMeasure2D01(torch.clamp(landmarks, max=1))
    heatmap_sum = heatmapper.forward(measure.coord).sum(1, keepdim=True).detach()

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/cycle_loss.json")))

    fake, fake_latent = enc_dec.generate(heatmap_sum)
    fake_latent_pred = enc_dec.encode_latent(fake)

    gan_model_tuda.discriminator_train([real_img], [fake.detach()])
    (
            gan_model_tuda.generator_loss([real_img], [fake]) +
            l1_loss(fake_latent_pred, fake_latent) * coefs["style"]
    ).minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    hm_pred = hg.forward(real_img)["hm_sum"]
    hm_ref = heatmapper.forward(landmarks).detach().sum(1, keepdim=True)
    gan_model_obratno.discriminator_train([hm_ref], [hm_pred.detach()])
    gan_model_obratno.generator_loss([hm_ref], [hm_pred]).__mul__(coefs["obratno"]) \
        .minimize_step(gan_model_obratno.optimizer.opt_min)

    fake2, _ = enc_dec.generate(heatmap_sum)
    WR.writable("cycle", mes_loss.forward)(hg.forward(fake2)["mes"], UniformMeasure2D01(landmarks)).__mul__(coefs["hm"]) \
        .minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    latent = enc_dec.encode_latent(g_transforms(image=real_img)["image"])
    restored = enc_dec.decode(hg.forward(real_img)["hm_sum"], latent)
    WR.writable("cycle2", psp_loss.forward)(real_img, real_img, restored, latent).__mul__(coefs["img"]) \
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
            f'{Paths.default.models()}/cardio_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl = verka_cardio_w2(hg)
            writer.add_scalar("verka", tl, i)

            sk_pred = hg.forward(test_img)["hm_sum"]
            fake, _ = enc_dec.generate(test_hm, test_noise)
            latent = enc_dec.encode_latent(test_img)
            restored = enc_dec.decode(sk_pred, latent)

            send_images_to_tensorboard(writer, fake + test_hm, "FAKE", i)
            send_images_to_tensorboard(writer, test_img + sk_pred, "REAL", i)
            send_images_to_tensorboard(writer, restored + sk_pred, "RESTORED", i)

