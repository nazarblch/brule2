import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from dataset.probmeasure import UniformMeasure2D01, UniformMeasure2DFactory
from dataset.toheatmap import heatmap_to_measure, ToGaussHeatMap
from gan.discriminator import Discriminator
from nn.common.view import View


def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dist, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1

def accuracy(output, target, idxs=None, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    if idxs is None:
        idxs = list(range(target.shape[-3]))
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]], thr=thr)
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc

def final_preds_untransformed(output, res):
    coords = get_preds(output) # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]], device=output.device)
                coords[n][p] += diff.sign() * .25
    coords += 0.5

    if coords.dim() < 3:
        coords = coords.unsqueeze(0)

    coords -= 1  # Convert from 1-based to 0-based coordinates

    return coords



class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        NormClass = nn.BatchNorm2d

        self.bn1 = NormClass(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = NormClass(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = NormClass(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, in_nc=3):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(in_nc, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'], in_nc=kwargs['in_nc'])
    return model


def _hg(arch, pretrained, progress, **kwargs):
    model = hg(**kwargs)
    return model


def hg1(pretrained=False, progress=True, num_blocks=1, num_classes=16, in_nc=3):
    return _hg('hg1', pretrained, progress, num_stacks=1, num_blocks=num_blocks,
               num_classes=num_classes, in_nc=in_nc)


def hg2(pretrained=False, progress=True, num_blocks=1, num_classes=16, in_nc=3):
    return _hg('hg2', pretrained, progress, num_stacks=2, num_blocks=num_blocks,
               num_classes=num_classes, in_nc=in_nc)


def hg4(pretrained=False, progress=True, num_blocks=1, num_classes=16, in_nc=3):
    return _hg('hg4', pretrained, progress, num_stacks=4, num_blocks=num_blocks,
               num_classes=num_classes, in_nc=in_nc)


def hg8(pretrained=False, progress=True, num_blocks=1, num_classes=16, in_nc=3):
    return _hg('hg8', pretrained, progress, num_stacks=8, num_blocks=num_blocks,
               num_classes=num_classes, in_nc=in_nc)


class HG_softmax2020(nn.Module):

    def __init__(self, num_classes=68, heatmap_size=64):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_size = heatmap_size
        self.model = hg2(num_classes=self.num_classes, num_blocks=1)
        self.up = nn.Upsample(size=256)

    def forward(self, image: Tensor):
        B, C, D, D = image.shape
        heatmaps: List[Tensor] = self.model.forward(image)
        out = heatmaps[-1]

        hm = self.up(self.postproc(out))
        coords, _ = heatmap_to_measure(hm)

        return {
            "out": out,
            "mes": UniformMeasure2D01(coords),
            'hm': hm,
            "softmax": self.postproc(out)
        }

    def postproc(self, hm: Tensor):

        B = hm.shape[0]

        return hm.clamp(-100, 30)\
                   .view(B, self.num_classes, -1)\
                   .softmax(dim=2)\
                   .view(B, self.num_classes, self.heatmap_size, self.heatmap_size) / self.num_classes


    def get_heatmaps(self, image: Tensor):
        B, C, D, D = image.shape
        return self.forward(image).clamp(-100, 30)\
                   .view(B, self.num_classes, -1)\
                   .softmax(dim=2)\
                   .view(B, self.num_classes, self.heatmap_size, self.heatmap_size) / self.num_classes

    def return_coords(self, image: Tensor):
        heatmaps = self.forward(image)
        coords, p = heatmap_to_measure(heatmaps)
        return coords


class HG_skeleton(nn.Module):

    def __init__(self, skeletoner, num_classes=68, heatmap_size=64):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_size = heatmap_size
        self.model = hg2(num_classes=self.num_classes, num_blocks=1)

        NormClass = nn.BatchNorm2d

        self.hm_to_coord = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 4, 2, 1),
            NormClass(num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_classes, num_classes, 4, 2, 1),
            NormClass(num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_classes, num_classes, 4, 2, 1),
            NormClass(num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_classes, num_classes, 4, 2, 1),
            NormClass(num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            View(num_classes * 4 * 4),
            nn.Linear(num_classes * 4 * 4, num_classes * 2),
            nn.Sigmoid(),
            View(num_classes, 2)
        )

        self.skeletoner = skeletoner

    def forward(self, image: Tensor):
        B, C, D, D = image.shape
        heatmaps: List[Tensor] = self.model.forward(image)
        out = heatmaps[-1]

        coords = self.hm_to_coord(out)
        sk = self.skeletoner.forward(coords).sum(dim=1, keepdim=True)

        assert coords.max().item() is not None
        assert coords.max().item() < 2

        return {
            "mes": UniformMeasure2D01(coords),
            "skeleton": sk
        }


class HG_heatmap(nn.Module):

    def __init__(self, heatmapper, num_classes=68, heatmap_size=64, image_size=256, num_blocks=1):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_size = heatmap_size
        self.model = hg2(num_classes=self.num_classes, num_blocks=num_blocks)

        NormClass = nn.BatchNorm2d

        self.hm_to_coord = []

        num_convs = int(math.log(image_size // 4, 2)) - 2
        for _ in range(num_convs):
            self.hm_to_coord += [
                nn.Conv2d(num_classes, num_classes, 4, 2, 1),
                NormClass(num_classes),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.hm_to_coord += [
            View(num_classes * 4 * 4),
            nn.Linear(num_classes * 4 * 4, num_classes * 2),
            nn.Sigmoid(),
            View(num_classes, 2)
        ]

        self.hm_to_coord = nn.Sequential(*self.hm_to_coord)

        self.heatmapper = heatmapper
        self.up = nn.Upsample(size=image_size)

    def postproc(self, hm: Tensor):

        B = hm.shape[0]

        return hm.clamp(-100, 30)\
                   .view(B, self.num_classes, -1)\
                   .softmax(dim=2)\
                   .view(B, self.num_classes, self.heatmap_size, self.heatmap_size)


    def forward(self, image: Tensor):
        B, C, D, D = image.shape
        heatmaps: List[Tensor] = self.model.forward(image)
        out = heatmaps[-1]

        coords = self.hm_to_coord(out)
        hm = self.heatmapper.forward(coords)

        # hm = self.up(self.postproc(out))
        # coords, _ = heatmap_to_measure(hm)
        # hm_g = self.heatmapper.forward(coords)

        assert coords.max().item() is not None
        assert coords.max().item() < 2

        return {
            "mes": UniformMeasure2D01(coords),
            "hm": hm,
            "hm_sum": hm.sum(dim=1, keepdim=True),
            "hm_g": None,
            "hm_g_sum": None,
        }


if __name__ == '__main__':

    model = hg2(num_classes=68)

    heatmaps = model.forward(torch.randn(4, 3, 256, 256))
    print(heatmaps[-1].shape)

    coords = final_preds_untransformed(heatmaps[-1], (64, 64))

    print(coords.shape)

