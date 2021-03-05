#%%
import json
import math
import sys, os

from models.autoencoder import StyleGanAutoEncoder

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))

from loss.l1 import l1_loss
from loss.mes import MesBceWasLoss, MesBceL2Loss
from metrics.board import send_images_to_tensorboard

from loss.weighted_was import OTWasLoss
from metrics.landmarks import verka_300w, verka_300w_w2
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


manualSeed = 73
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 8
image_size = 256
noise_size = 512
n_mlp = 8
lr_mlp = 0.01

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
W300DatasetLoader.batch_size = batch_size
W300Landmarks.batch_size = batch_size
Celeba.batch_size = batch_size

starting_model_number = 560000 + 10000
weights = torch.load(
    f'{Paths.default.models()}/300w_encoder_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

enc_dec = StyleGanAutoEncoder(weights)

discriminator_img = Discriminator(image_size)
discriminator_img.load_state_dict(weights['di'])
discriminator_img = discriminator_img.cuda()

gan_model_tuda = StyleGanModel[HeatmapToImage](enc_dec.generator, StyleGANLoss(discriminator_img), (0.001/4, 0.0015/4))

style_opt = optim.Adam(enc_dec.style_encoder.parameters(), lr=2e-5)

writer = SummaryWriter(f"{Paths.default.board()}/hm2img{int(time.time())}")
WR.writer = writer

N = 3148

heatmapper = ToGaussHeatMap(256, 4)

batch = next(LazyLoader.w300().loader_train_inf)
test_img = batch["data"].cuda()
test_landmarks = torch.clamp(batch["meta"]["keypts_normalized"].cuda(), max=1).cuda()
test_hm = heatmapper.forward(test_landmarks).sum(1, keepdim=True).detach()
test_noise = mixing_noise(batch_size, 512, 0.9, device)

psp_loss = PSPLoss(l2_lambda=2.0).cuda()

image_accumulator = Accumulator(enc_dec.generator, decay=0.99, write_every=100)
enc_accumulator = Accumulator(enc_dec.style_encoder, decay=0.99, write_every=100)

for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.w300().loader_train_inf)

    real_img = next(LazyLoader.w300().loader_train_inf)["data"].cuda()
    landmarks = torch.clamp(batch["meta"]["keypts_normalized"].cuda(), max=1).cuda()
    heatmap_sum = heatmapper.forward(landmarks).sum(1, keepdim=True).detach()

    fake, fake_latent = enc_dec.generate(heatmap_sum)
    fake_latent_pred = enc_dec.encode_latent(fake)

    real_gan_img = real_img if i % 2 == 0 else next(LazyLoader.celeba().loader).cuda()

    gan_model_tuda.discriminator_train([real_gan_img], [fake.detach()])
    (
        gan_model_tuda.generator_loss([real_gan_img], [fake]) +
        l1_loss(fake_latent_pred, fake_latent)
    ).minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    latent = enc_dec.encode_latent(real_img)
    restored = enc_dec.decode(heatmap_sum, latent)
    WR.writable("cycle2", psp_loss.forward)(real_img, real_img, restored, latent).__mul__(20)\
        .minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    image_accumulator.step(i)
    # enc_accumulator.step(i)

    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'gi': enc_dec.generator.state_dict(),
                'di': discriminator_img.state_dict(),
                's': enc_dec.style_encoder.state_dict()
            },
            f'{Paths.default.models()}/300w_encoder_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            fake, _ = enc_dec.generate(test_hm, test_noise)
            latent = enc_dec.encode_latent(test_img)
            restored = enc_dec.decode(test_hm, latent)

            send_images_to_tensorboard(writer, fake + test_hm, "FAKE", i)
            send_images_to_tensorboard(writer, test_img + test_hm, "REAL", i)
            send_images_to_tensorboard(writer, restored + test_hm, "RESTORED", i)

