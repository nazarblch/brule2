#%%
import argparse
import json
import math
import sys, os


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))

from argparse import ArgumentParser
from metrics.segm import verka_segm
import albumentations

from dataset.transforms import ToNumpy, NumpyBatch, ToTensor

import torch.nn.functional as F
from parameters.dataset import DatasetParameters
from parameters.model import ModelParameters
from parameters.run import RuntimeParameters
from models.autoencoder import StyleGanAutoEncoder
from loss.l1 import l1_loss
from loss.mes import MesBceWasLoss, MesBceL2Loss
from metrics.board import send_images_to_tensorboard

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
import segmentation_models_pytorch as smp
from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, W300Landmarks, BraTSLoader
from dataset.probmeasure import UniformMeasure2D01
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel, CondStyleGanModel
from gan.nn.stylegan.discriminator import Discriminator, ConditionalDiscriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths


class LossBinaryDice(nn.Module):
    def __init__(self, dice_weight=2):
        super(LossBinaryDice, self).__init__()
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        targets = targets.squeeze().float()
        outputs = outputs.squeeze().float()
        loss = self.nll_loss(outputs, targets)

        if self.dice_weight:
            smooth = torch.tensor(1e-15).float()
            target = (targets > 1e-10).float()
            prediction = outputs
            dice_part = (1 - (2 * torch.sum(prediction * target, dim=(1,2)) + smooth) / \
                         (torch.sum(prediction, dim=(1,2)) + torch.sum(target, dim=(1,2)) + smooth))

            loss += self.dice_weight * dice_part.mean()
        return Loss(loss)


manualSeed = 73
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
BraTSLoader.batch_size = batch_size

g_transforms = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            albumentations.ElasticTransform(p=0.3, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=15),
        ])),
        ToTensor(device),
    ])

starting_model_number = 20000 + 90000 + 60000
weights = torch.load(
    f'{Paths.default.models()}/BraTS_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

enc_dec = StyleGanAutoEncoder().load_state_dict(weights).cuda()

discriminator_img = Discriminator(image_size, input_nc=3)
discriminator_img.load_state_dict(weights['di'])
discriminator_img = discriminator_img.cuda()

hg = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
hg = nn.Sequential(
    hg, nn.Sigmoid()
)
hg.load_state_dict(weights['gh'])

hg = hg.cuda()
hm_discriminator = Discriminator(image_size, input_nc=1, channel_multiplier=1)
hm_discriminator.load_state_dict(weights["dh"])
hm_discriminator = hm_discriminator.cuda()

gan_model_tuda = StyleGanModel[HeatmapToImage](enc_dec.generator, StyleGANLoss(discriminator_img), (0.001/4, 0.0015/4))
gan_model_obratno = StyleGanModel[HG_skeleton](hg, StyleGANLoss(hm_discriminator), (2e-5, 0.0015/4))

style_opt = optim.Adam(enc_dec.style_encoder.parameters(), lr=1e-5)

writer = SummaryWriter(f"{Paths.default.board()}/BraTS{int(time.time())}")
WR.writer = writer

brats_data = BraTSLoader()
loader = brats_data.loader_train_inf
batch = next(loader)
test_img = batch[0].cuda()
# test_seg = batch[1].cuda()
test_seg = next(brats_data.loader_masks_bc_inf).cuda()

test_noise = mixing_noise(batch_size, 512, 0.9, device)

psp_loss = PSPLoss(id_lambda=0).cuda()
# mes_loss = MesBceWasLoss(heatmapper, bce_coef=100000, was_coef=100)

our_loss = LossBinaryDice(dice_weight=2)

image_accumulator = Accumulator(enc_dec.generator, decay=0.99, write_every=100)
hm_accumulator = Accumulator(hg, decay=0.99, write_every=100)


for i in range(200001):

    WR.counter.update(i)

    batch = next(loader)
    real_img = batch[0].cuda()
    # real_seg = batch[1].cuda()
    real_seg = next(brats_data.loader_masks_bc_inf).cuda()

    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/cycle_loss.json")))

    fake, fake_latent = enc_dec.generate(real_seg)
    fake_latent_pred = enc_dec.encode_latent(fake)

    gan_model_tuda.discriminator_train([real_img], [fake.detach()])
    (
        gan_model_tuda.generator_loss([real_img], [fake]) +
        l1_loss(fake_latent_pred, fake_latent) * coefs["style"]
    ).minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    seg_pred = hg.forward(real_img)
    gan_model_obratno.discriminator_train([real_seg], [seg_pred.detach()])
    gan_model_obratno.generator_loss([real_seg], [seg_pred]).__mul__(coefs["obratno"])\
        .minimize_step(gan_model_obratno.optimizer.opt_min)

    fake2, _ = enc_dec.generate(real_seg)
    WR.writable("cycle", our_loss.forward)(hg.forward(fake2), real_seg).__mul__(coefs["hm"])\
        .minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    latent = enc_dec.encode_latent(real_img)
    # latent = enc_dec.encode_latent(g_transforms(image=real_img)["image"])
    restored = enc_dec.decode(hg.forward(real_img), latent)
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
            f'{Paths.default.models()}/BraTS_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl2 = verka_segm(hg, brats_data)
            writer.add_scalar("verka", tl2, i)

            sk_pred = hg.forward(test_img)
            fake, _ = enc_dec.generate(test_seg, test_noise)
            latent = enc_dec.encode_latent(test_img)
            restored = enc_dec.decode(sk_pred, latent)

            send_images_to_tensorboard(writer, fake, "FAKE", i)
            send_images_to_tensorboard(writer, test_img, "REAL", i)
            send_images_to_tensorboard(writer, restored, "RESTORED", i)
            send_images_to_tensorboard(writer, sk_pred, "HM", i)
            send_images_to_tensorboard(writer, test_seg, "HM REAL", i)

