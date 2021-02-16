import json

import torch
from torch import nn, Tensor

from dataset.probmeasure import UniformMeasure2D01, ProbabilityMeasureFabric, UniformMeasure2DFactory
from dataset.toheatmap import ToGaussHeatMap, CoordToGaussSkeleton
from gan.loss.loss_base import Loss
from gan.noise.stylegan import mixing_noise
from wr import WR

def noviy_hm_loss(pred, target, coef=1.0):

    pred = pred / pred.sum(dim=[2, 3], keepdim=True).detach()
    target = target / target.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred, target) * coef
    )

def coord_hm_loss(pred_coord: Tensor, target_hm: Tensor, coef=1.0):
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True)
    target_hm = target_hm.detach()

    heatmapper = ToGaussHeatMap(256, 1)

    target_coord = UniformMeasure2DFactory.from_heatmap(target_hm).coord.detach()
    sk = CoordToGaussSkeleton(target_hm.shape[-1], 1)
    pred_sk = sk.forward(pred_coord).sum(dim=1, keepdim=True)
    target_sk = sk.forward(target_coord).sum(dim=1, keepdim=True).detach()
    pred_hm = heatmapper.forward(pred_coord).sum(dim=1, keepdim=True)
    pred_hm = pred_hm / pred_hm.sum(dim=[2, 3], keepdim=True).detach()
    target_hm = heatmapper.forward(target_coord).sum(dim=1, keepdim=True).detach()
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred_hm, target_hm) * coef * 0.5 +
        noviy_hm_loss(pred_sk, target_sk, coef).to_tensor() * 0.5 +
        nn.MSELoss()(pred_coord, target_coord) * (0.001 * coef) +
        nn.L1Loss()(pred_coord, target_coord) * (0.001 * coef)
    )


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def gan_trainer(model, generator, decoder, style_encoder, style_opt):

    def gan_train(real_img, skeleton):

        B = real_img.shape[0]
        C = 512

        requires_grad(generator, True)
        requires_grad(decoder, True)
        condition = skeleton.detach().requires_grad_(True)

        noise = mixing_noise(B, C, 0.9, real_img.device)

        fake, fake_latent = generator(condition, noise, return_latents=True)

        model.discriminator_train([real_img], [fake], [condition])

        WR.writable("Generator loss", model.generator_loss)([real_img], [fake], [condition]) \
            .minimize_step(model.optimizer.opt_min)

        fake = fake.detach()

        fake_latent_pred = style_encoder(fake)
        restored = decoder(condition, style_encoder(real_img))
        fake_latent = torch.cat([f[:, None, :] for f in fake_latent], dim=1).detach()

        coefs = json.load(open("../parameters/gan_loss.json"))

        (
            WR.L1("L1 restored")(restored, real_img) * coefs["L1 restored"] +
            WR.L1("L1 style gan")(fake_latent_pred, fake_latent) * coefs["L1 style gan"]
        ).minimize_step(
            model.optimizer.opt_min,
            style_opt
        )

    return gan_train


def content_trainer_with_gan(encoder_HG, model, generator, decoder, style_encoder):

    C = 512
    heatmapper = ToGaussHeatMap(256, 1)

    def do_train(real_img):

        B = real_img.shape[0]

        coefs = json.load(open("../parameters/content_loss.json"))

        requires_grad(encoder_HG, True)
        requires_grad(decoder, False)
        requires_grad(generator, False)

        encoded = encoder_HG(real_img)
        pred_measures: UniformMeasure2D01 = encoded["mes"]

        heatmap_content = heatmapper.forward(pred_measures.coord).detach()

        restored = decoder(encoded["skeleton"], style_encoder(real_img))

        noise = mixing_noise(B, C, 0.9, real_img.device)
        fake, _ = generator(encoded["skeleton"], noise)
        fake_content = encoder_HG(fake.detach())["mes"]

        ll = (
                WR.L1("L1 image")(restored, real_img) * coefs["L1 image"] +
                WR.writable("fake_content loss", coord_hm_loss)(
                    fake_content, heatmap_content
                ) * coefs["fake_content loss"] +
                WR.writable("Fake-content D", model.loss.generator_loss)(
                    real=None,
                    fake=[fake, encoded["skeleton"].detach()]) * coefs["Fake-content D"]
        )

        ll.minimize_step(model.optimizer.opt_min)

    return do_train



