
import random
import time

from typing import List
import torch.utils.data
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from gan.nn.stylegan.discriminator import Discriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator, HeatmapToImage
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths


def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)


manualSeed = 71
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 16
image_size = 256
noise_size = 512

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

test_sample_z = torch.randn(8, noise_size, device=device)
Celeba.batch_size = batch_size

generator = Generator(FromStyleConditionalGenerator(image_size, noise_size), n_mlp=8)
discriminator = Discriminator(image_size)

starting_model_number = 290000
weights = torch.load(
    f'{Paths.default.models()}/celeba_gan_256_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

discriminator.load_state_dict(weights['d'])
generator.load_state_dict(weights['g'])

heatmap2image = HeatmapToImage(generator.gen, generator.z_to_style, 1).cuda()

hm = torch.randn(8, 1, 256, 256, device=device)

heatmap2image.forward(hm, [test_sample_z])
