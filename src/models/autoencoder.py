import torch

from gan.nn.stylegan.generator import HeatmapToImage, FromStyleConditionalGenerator, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.nn.stylegan.style_encoder import GradualStyleEncoder, StyleEncoder
from torch import Tensor

from gan.noise.stylegan import mixing_noise


class StyleGanAutoEncoder:

    def __init__(self, image_size: int = 256, noise_size: int = 512, hm_nc=1):

        style_count = {
            64: 10,
            128: 12,
            256: 14,
            512: 16,
            1024: 18,
        }

        self.noise_size = noise_size

        self.generator = HeatmapToImage(
            FromStyleConditionalGenerator(image_size, noise_size),
            NoiseToStyle(512, 8, 0.01, style_count[image_size]),
            hm_nc
        )

        self.decoder = HeatmapAndStyleToImage(self.generator)
        self.style_encoder = GradualStyleEncoder(50, 3, mode="ir", style_count=style_count[image_size])


    def load_state_dict(self, weights):
        self.style_encoder.load_state_dict(weights["s"])
        # print("commented gen load")
        self.generator.load_state_dict(weights['gi'])
        return self

    def cuda(self):
        self.generator = self.generator.cuda()
        self.decoder = HeatmapAndStyleToImage(self.generator)
        self.style_encoder = self.style_encoder.cuda()
        return self

    def encode_latent(self, image: Tensor) -> Tensor:
        return self.style_encoder(image)

    def generate(self, one_channel_heatmap: Tensor, noise=None) -> (Tensor, Tensor):
        # assert one_channel_heatmap.shape[1] == 1
        if noise is None:
            noise = mixing_noise(one_channel_heatmap.shape[0], self.noise_size, 0.9, one_channel_heatmap.device)
        fake, fake_latent = self.generator.forward(one_channel_heatmap, noise, return_latents=True)
        fake_latent = torch.cat([f[:, None, :] for f in fake_latent], dim=1).detach()
        return fake, fake_latent

    def decode(self, one_channel_heatmap: Tensor, latent: Tensor):
        # assert one_channel_heatmap.shape[1] == 1
        return self.decoder(one_channel_heatmap, latent)




