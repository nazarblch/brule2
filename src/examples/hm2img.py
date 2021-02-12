#%%
import math
import sys, os


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))

from wr import WR

import random
from models.hg import HG_skeleton, HGDiscriminator, HG_skeleton_2
from gan.loss.loss_base import Loss

import time
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton

from typing import List
import torch.utils.data
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader
from dataset.probmeasure import UniformMeasure2D01
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from gan.nn.stylegan.discriminator import Discriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage, NoiseToStyle
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths

def verka(enc):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        pred = enc(data)["mes"].coord
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((pred - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    print("test loss: ", sum_loss / len(LazyLoader.w300().test_dataset))
    return sum_loss / len(LazyLoader.w300().test_dataset)


def sup_loss(pred_hm, target_hm, pred_mes, target_mes):

    pred_hm = pred_hm / (pred_hm.sum(dim=[2, 3], keepdim=True).detach() + 1e-8)
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred_hm, target_hm) * 100000 +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * 500
    )


def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)


manualSeed = 71
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


starting_model_number = 360000
weights = torch.load(
    f'{Paths.default.models()}/hm2img_{str(starting_model_number).zfill(6)}.pt',
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

heatmapper = ToGaussHeatMap(64, 2)
skeletoner = CoordToGaussSkeleton(256, 4)
hg = HG_skeleton_2(skeletoner).cuda()
hm_discriminator = Discriminator(image_size, input_nc=1).cuda()

gan_model_tuda = StyleGanModel[HeatmapToImage](heatmap2image, StyleGANLoss(discriminator_img), (0.001, 0.0015))
gan_model_obratno = StyleGanModel[HG_skeleton](hg, StyleGANLoss(hm_discriminator), (2e-5, 0.0015))

writer = SummaryWriter(f"{Paths.default.board()}/hm2img{int(time.time())}")
WR.writer = writer

#%%

for i in range(100000):

    WR.counter.update(i)

    batch = next(LazyLoader.w300().loader_train_inf)

    real_img = batch["data"].cuda()
    landmarks = batch["meta"]["keypts_normalized"].cuda()
    measure = UniformMeasure2D01(torch.clamp(landmarks, max=1))
    skeleton = skeletoner.forward(measure.coord).sum(1, keepdim=True).detach()
    noise = mixing_noise(batch_size, 512, 0.9, device)

    #%%

    fake, _ = heatmap2image.forward(skeleton, noise)

    gan_model_tuda.discriminator_train([real_img], [fake.detach()])
    gan_model_tuda.generator_loss([real_img], [fake]).minimize_step(gan_model_tuda.optimizer.opt_min)

    #%%

    sk_pred = hg.forward(real_img)["skeleton"]
    gan_model_obratno.discriminator_train([skeleton], [sk_pred.detach()])
    gan_model_obratno.generator_loss([skeleton], [sk_pred]).minimize_step(gan_model_obratno.optimizer.opt_min)

    fake2, _ = heatmap2image.forward(skeleton, noise)
    pred2 = hg.forward(fake2)
    hm_ref = heatmapper.forward(measure.coord)
    WR.writable("cycle", sup_loss)(pred2["hm"], hm_ref, pred2["mes"], measure).minimize_step(gan_model_tuda.optimizer.opt_min, gan_model_obratno.optimizer.opt_min)

    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'gi': heatmap2image.state_dict(),
                'gh': hg.state_dict(),
                'di': discriminator_img.state_dict(),
                'dh': hm_discriminator.state_dict()
            },
            f'{Paths.default.models()}/hm2img_{str(i + starting_model_number).zfill(6)}.pt',
        )

    if i % 100 == 0:
        print(i)
        with torch.no_grad():

            tl = verka(hg)
            writer.add_scalar("verka", tl, i)

            sk_pred = hg.forward(real_img)["skeleton"]
            fake, _ = heatmap2image.forward(skeleton, noise)
            send_images_to_tensorboard(writer, fake + skeleton, "FAKE", i)
            send_images_to_tensorboard(writer, real_img + sk_pred, "REAL", i)

