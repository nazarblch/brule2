from typing import List
import torch.utils.data
from torch import Tensor, optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


def plot_img_with_lm(img: torch.Tensor, lm: torch.Tensor, nrows=4, ncols=4):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.5, nrows * 2.5))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            print(index)
            axes[i, j].axis("off")
            axes[i, j].imshow(((img[index]).permute(1, 2, 0).numpy() + 1) * 0.5)
            b = lm[index].numpy() * 255
            axes[i, j].scatter(b[:, 0], b[:, 1], c="white", s=2)

    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


def send_images_to_tensorboard(writer: SummaryWriter,
                               data: Tensor,
                               name: str,
                               iter: int,
                               count=8,
                               normalize=True,
                               range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)