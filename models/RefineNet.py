import torch.nn as nn
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from collections import OrderedDict
import math
from argparse import Namespace


class ResNet(nn.Module):
    def __init__(self, in_channels: int, k: int) -> None:
        super(ResNet, self).__init__()
        pad = math.floor((k - 1) / 2)
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k, 1, pad),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, k, 1, pad),
            nn.BatchNorm2d(in_channels),
        )
        self.avg = nn.AvgPool2d((1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.res(x) + self.avg(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, n: int) -> None:
        super(Upsample, self).__init__()
        k = 3
        pad = math.floor((k - 1) / 2)
        od = OrderedDict()

        for i in range(n):
            od[f'conv{i}'] = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, pad)
            od[f'bn{i}'] = nn.BatchNorm2d(in_channels)
            od[f'actv{i}'] = nn.ReLU(inplace=True)

        od[f'conv{i+1}'] = nn.Conv2d(in_channels, in_channels, k, 1, pad)
        od[f'bn{i+1}'] = nn.BatchNorm2d(in_channels)
        od[f'actv{i+1}'] = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(od)

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, n: int) -> None:
        super(Downsample, self).__init__()
        pad = math.floor((k - 1) / 2)
        od = OrderedDict()
        od['rp'] = nn.ReflectionPad2d(pad)
        od['conv0'] = nn.Conv2d(in_channels, out_channels, k, 1, 0)
        od['bn0'] = nn.BatchNorm2d(out_channels)
        od['actv0'] = nn.ReLU(inplace=True)

        for i in range(n):
            od[f'conv{i+1}'] = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            od[f'bn{i+1}'] = nn.BatchNorm2d(out_channels)
            od[f'actv{i+1}'] = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(od)

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(x)


class Normalize(nn.Module):
    def __init__(self) -> None:
        super(Normalize, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.ratio = 1/512

    def forward(self, x: List[Tensor]) -> Tensor:
        x[1] = x[1] * self.ratio
        x[2] = self.sm(x[2])
        x = torch.cat(x, dim=1)
        return x


class RefineNet(nn.Module):
    def __init__(self, opt: Namespace) -> None:
        super(RefineNet, self).__init__()
        self.opt = opt
        self.layers = 12
        self.n = 64
        self.k = 3
        self.in_channels = 8
        self.factor = 3

        r = [ResNet(self.n, self.k) for _ in range(math.floor((self.layers-2) / 2))]
        
        self.res = nn.Sequential(*r)

        self.normalize = Normalize()

        self.up1 = Upsample(self.n, self.factor)
        self.up2 = Upsample(self.n, self.factor)
        self.up3 = Upsample(self.n, self.factor)

        self.down = Downsample(self.in_channels, self.n, 9, self.factor)

        self.conv1 = nn.Conv2d(self.n+2, 2, 3, 1, 1) # flow
        self.conv2 = nn.Conv2d(self.n+2, 2, 3, 1, 1) # mask
        self.conv3 = nn.Conv2d(self.n+1, 1, 3, 1, 1) # rho

    def forward(self, x: List[Tensor]) -> Tensor:
        # x = [img:3, flow:2, mask:2, rho:1]
        normalized_x = self.normalize(x)

        downsampled_x = self.down(normalized_x)

        res = self.res(downsampled_x)

        f_flow = self.up1(res)
        f_mask = self.up2(res)
        f_rho = self.up3(res)

        flow = self.conv1(torch.cat([f_flow, x[1]], dim=1))
        mask = self.conv2(torch.cat([f_mask, x[2]], dim=1))
        rho = self.conv3(torch.cat([f_rho, x[3]], dim=1))

        return [flow, mask, rho]
