from typing import List
import torch.utils.data
from torch import Tensor, optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


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