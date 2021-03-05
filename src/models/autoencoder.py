import torch

from gan.nn.stylegan.generator import HeatmapToImage, FromStyleConditionalGenerator, NoiseToStyle, \
    HeatmapAndStyleToImage
from gan.nn.stylegan.style_encoder import GradualStyleEncoder
from torch import Tensor

from gan.noise.stylegan import mixing_noise


class StyleGanAutoEncoder:

    def __init__(self, weights: dict, image_size: int = 256, noise_size: int = 512):

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
            NoiseToStyle(512, 8, 0.01, 14),
            1
        )

        self.generator.load_state_dict(weights['gi'])
        self.generator = self.generator.cuda()
        self.decoder = HeatmapAndStyleToImage(self.generator)

        self.style_encoder = GradualStyleEncoder(50, 3, style_count=style_count[image_size])
        self.style_encoder.load_state_dict(weights["s"])
        self.style_encoder = self.style_encoder.cuda()

    def encode_latent(self, image: Tensor) -> Tensor:
        return self.style_encoder(image)

    def generate(self, one_channel_heatmap: Tensor, noise=None) -> (Tensor, Tensor):
        assert one_channel_heatmap.shape[1] == 1
        if noise is None:
            noise = mixing_noise(one_channel_heatmap.shape[0], self.noise_size, 0.9, one_channel_heatmap.device)
        fake, fake_latent = self.generator.forward(one_channel_heatmap, noise, return_latents=True)
        fake_latent = torch.cat([f[:, None, :] for f in fake_latent], dim=1).detach()
        return fake, fake_latent

    def decode(self, one_channel_heatmap: Tensor, latent: Tensor):
        assert one_channel_heatmap.shape[1] == 1
        return self.decoder(one_channel_heatmap, latent)




