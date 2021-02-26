#%%
import json
import math
import sys, os


sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/src'))
sys.path.append(os.path.join(sys.path[0], '/home/nazar/PycharmProjects/brule2/gans/'))

from gan.loss.perceptual.psp import PSPLoss
from train_procedure import gan_trainer, content_trainer_with_gan, coord_hm_loss


from models.hg import HG_skeleton

from wr import WR

from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from torch import optim

import random
from gan.loss.loss_base import Loss

import time
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton

from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader, Cardio
from dataset.probmeasure import UniformMeasure2D01
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel, CondStyleGanModel
from gan.nn.stylegan.discriminator import Discriminator, ConditionalDiscriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths

def verka(enc):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.cardio().test_loader):
        data = batch['image'].to(device)
        landmarks = batch["keypoints"].cuda()
        pred = enc(data)["mes"].coord
        sum_loss += ((pred - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1)).sum().item()
    print("test loss: ", sum_loss / len(LazyLoader.cardio().test_dataset))
    return sum_loss / len(LazyLoader.cardio().test_dataset)


def sup_loss(pred_mes, target_mes):
    pred_hm = heatmapper.forward(pred_mes.coord)
    target_hm = heatmapper.forward(target_mes.coord)

    pred_hm = pred_hm / (pred_hm.sum(dim=[1, 2, 3], keepdim=True).detach() + 1e-8)
    target_hm = target_hm / target_hm.sum(dim=[1, 2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred_hm, target_hm) * 1000000 +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * 20000
    )

def l1_loss(pred, target):
    return Loss(nn.L1Loss().forward(pred, target))

def l2_loss(pred, target):
    return Loss(nn.MSELoss().forward(pred, target))

def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)


manualSeed = 71
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 4
image_size = 256
noise_size = 512
n_mlp = 8
lr_mlp = 0.01

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

Cardio.batch_size = batch_size


starting_model_number = 110000
weights = torch.load(
    f'{Paths.default.models()}/cardio_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

discriminator_img = Discriminator(image_size)
discriminator_img.load_state_dict(weights['di'])
discriminator_img = discriminator_img.cuda()

heatmap2image = HeatmapToImage(
    FromStyleConditionalGenerator(image_size, noise_size),
    NoiseToStyle(512, n_mlp, lr_mlp, 14),
    1
)
heatmap2image.load_state_dict(weights['gi'])
heatmap2image = heatmap2image.cuda()

heatmapper = ToGaussHeatMap(256, 4)
# heatmapper_big = ToGaussHeatMap(256, 10)
skeletoner = CoordToGaussSkeleton(256, 4)
hg = HG_skeleton(skeletoner, num_classes=200)
# hg.load_state_dict(weights['gh'])
hg = hg.cuda()
hm_discriminator = Discriminator(image_size, input_nc=201, channel_multiplier=0.5)
# hm_discriminator.load_state_dict(weights["dh"])
hm_discriminator = hm_discriminator.cuda()

gan_model_tuda = StyleGanModel[HeatmapToImage](heatmap2image, StyleGANLoss(discriminator_img), (0.001/4, 0.0015/4))
gan_model_obratno = StyleGanModel[HG_skeleton](hg, StyleGANLoss(hm_discriminator), (2e-5, 0.0002))

style_encoder = GradualStyleEncoder(50, 3, style_count=14)
style_encoder.load_state_dict(weights["s"])
style_encoder = style_encoder.cuda()
decoder = HeatmapAndStyleToImage(heatmap2image)
style_opt = optim.Adam(style_encoder.parameters(), lr=1e-4)

writer = SummaryWriter(f"{Paths.default.board()}/cardio{int(time.time())}")
WR.writer = writer

#%%
lm = next(LazyLoader.cardio_landmarks().loader_train_inf)

batch = next(LazyLoader.cardio().loader_train_inf)
test_img = batch["image"].cuda()
test_landmarks = batch["keypoints"].cuda()
test_measure = UniformMeasure2D01(torch.clamp(test_landmarks, max=1))
# test_skeleton = skeletoner.forward(test_measure.coord).sum(1, keepdim=True).detach()
test_hm = heatmapper.forward(test_measure.coord).sum(1, keepdim=True).detach()
test_noise = mixing_noise(batch_size, 512, 0.9, device)

psp_loss = PSPLoss(id_lambda=0).cuda()

image_accumulator = Accumulator(heatmap2image, decay=0.99, write_every=100)
hm_accumulator = Accumulator(hg, decay=0.99, write_every=100)

for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.cardio().loader_train_inf)

    real_img = batch["image"].cuda()
    landmarks = batch["keypoints"].cuda()
    measure = UniformMeasure2D01(torch.clamp(landmarks, max=1))
    # skeleton = skeletoner.forward(measure.coord).sum(1, keepdim=True).detach()
    heatmap = heatmapper.forward(measure.coord).sum(1, keepdim=True).detach()
    noise = mixing_noise(batch_size, 512, 0.9, device)

    #%%
    coefs = json.load(open("../parameters/cycle_loss.json"))

    fake, fake_latent = heatmap2image.forward(heatmap, noise, return_latents=True)
    fake_latent = torch.cat([f[:, None, :] for f in fake_latent], dim=1).detach()
    fake_latent_pred = style_encoder.forward(fake)

    gan_model_tuda.discriminator_train([real_img], [fake.detach()])
    (
        gan_model_tuda.generator_loss([real_img], [fake]) +
        l1_loss(fake_latent_pred, fake_latent) * coefs["style"]
    ).minimize_step(gan_model_tuda.optimizer.opt_min, style_opt)

    #%%

    hm_pred = heatmapper.forward(hg.forward(real_img)["mes"].coord)
    hm_pred = torch.cat([hm_pred, hm_pred.sum(dim=1, keepdim=True)], dim=1)
    hm_ref = heatmapper.forward(measure.coord).detach()
    hm_ref = torch.cat([hm_ref, hm_ref.sum(dim=1, keepdim=True)], dim=1)
    gan_model_obratno.discriminator_train([hm_ref], [hm_pred.detach()])
    gan_model_obratno.generator_loss([hm_ref], [hm_pred]).__mul__(coefs["obratno"]).minimize_step(gan_model_obratno.optimizer.opt_min)

    fake2, _ = heatmap2image.forward(heatmap, noise)
    pred2 = hg.forward(fake2)
    WR.writable("cycle", sup_loss)(pred2["mes"], measure).__mul__(coefs["hm"]).minimize_step(
        gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    sk_pred3 = heatmapper.forward(hg.forward(real_img[:4])["mes"].coord).sum(1, keepdim=True)
    latent = style_encoder.forward(real_img[:4])
    restored = decoder.forward(sk_pred3, latent)
    WR.writable("cycle2", psp_loss)(real_img[:4], real_img[:4], restored, latent).__mul__(coefs["img"]).minimize_step(
        gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min, style_opt)

    image_accumulator.step(i)
    hm_accumulator.step(i)

    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'gi': heatmap2image.state_dict(),
                'gh': hg.state_dict(),
                'di': discriminator_img.state_dict(),
                'dh': hm_discriminator.state_dict(),
                's': style_encoder.state_dict()
            },
            f'{Paths.default.models()}/cardio_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl = verka(hg)
            writer.add_scalar("verka", tl, i)

            sk_pred = heatmapper.forward(hg.forward(test_img)["mes"].coord).sum(1, keepdim=True)
            fake, _ = heatmap2image.forward(test_hm, test_noise)
            latent = style_encoder.forward(test_img)
            restored = decoder.forward(sk_pred, latent)

            send_images_to_tensorboard(writer, fake + test_hm, "FAKE", i)
            send_images_to_tensorboard(writer, test_img + sk_pred, "REAL", i)
            send_images_to_tensorboard(writer, restored + sk_pred, "RESTORED", i)

