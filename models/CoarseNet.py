import torch.nn as nn
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from collections import OrderedDict
import math
from argparse import Namespace


class CreateOutput(nn.Module):
    def __init__(self, in_channels: int, w: int, scale: int) -> None:
        super(CreateOutput, self).__init__()
        k = 3
        step = 1
        pad = math.floor((k - 1) / 2)
        self.ratio = w / 2 ** (scale-1)
        self.th = nn.Tanh()
        self.conv1_0 = nn.Conv2d(in_channels, 2, k, step, pad)
        self.conv1_1 = nn.Conv2d(in_channels, 2, k, step, pad)
        self.conv2 = nn.Conv2d(in_channels, 1, k, step, pad)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        if type(x) != Tensor:
            x = torch.cat(x, dim=1) 
        
        flow = self.conv1_0(x)
        flow = self.th(flow).clone()
        flow *= self.ratio

        mask = self.conv1_1(x)

        rho = self.conv2(x)

        return [flow, mask, rho]


class NormalizeOutput(nn.Module):
    def __init__(self, w: int, scale: int) -> None:
        super(NormalizeOutput, self).__init__()
        self.ratio = 2.0 ** (scale-1) / w
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x: List[Tensor]) -> Tensor:
        x[0] *= self.ratio
        x[1] = self.sm(x[1])

        x = torch.cat(x, dim=1)
        return self.up(x)
        

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, step: int) -> None:
        super(Encoder, self).__init__()
        pad = math.floor((k-1)/2)
        od = OrderedDict()
        od['conv'] = nn.Conv2d(in_channels, out_channels, k, step, pad)
        od['batch_norm'] = nn.BatchNorm2d(out_channels)
        od['actv'] = nn.ReLU(inplace=False)
        
        self.encoder = nn.Sequential(od)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, step: int) -> None:
        super(Decoder, self).__init__()
        pad = math.floor((k-1)/2)
        od = OrderedDict()
        od['conv'] = nn.Conv2d(in_channels, out_channels, k, step, pad)
        od['batch_norm'] = nn.BatchNorm2d(out_channels)
        od['actv'] = nn.ReLU(inplace=False)
        od['up'] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder = nn.Sequential(od)

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Tensor:
        if type(x) != Tensor:
            x = torch.cat(x, dim=1) 

        return self.decoder(x)


class RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super(RCAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Tanh(),
        )
		
    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(self.avg_pool(x))
        return x * y
	

class Residual_Block(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super(Residual_Block, self).__init__()
        k = 3
        step = 1
        pad = (k - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, k, step, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, k, step, pad),
            RCAB(in_channels, 16),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out += x
        return out
	
	
class RIRB(nn.Module):
    def __init__(self, growth_rate: int, layers: int) -> None:
        super(RIRB, self).__init__()
        k = 3
        step = 1
        pad = (k - 1) // 2
        residuals = []
        for _ in range(layers):
            residuals.append(Residual_Block(growth_rate, growth_rate))
        self.residuals = nn.Sequential(*residuals)
        self.conv = nn.Conv2d(growth_rate, growth_rate, k, step, pad)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.residuals(x)) + x
        return x


class CoarseNet(nn.Module):
    def __init__(self, opt: Namespace) -> None:
        super(CoarseNet, self).__init__()
        self.opt = opt
        c_in = 3
        w = 512

        c_0 = c_1 = 16
        c_2 = 32
        c_3 = 64
        c_4 = 128
        c_5 = c_6 = 256

        n_out = 3  # num of output branches (flow, mask, rho)
        c_out_num = 5  # total num of channels (2 + 2 + 1)

        self.encoder0 = nn.Sequential(
            Encoder(c_in, c_0, 3, 1),
            Encoder(c_0, c_0, 3, 1),
        )
        self.encoder1 = nn.Sequential(
            Encoder(c_0, c_1, 3, 2),
            Encoder(c_1, c_1, 3, 1),
        )
        self.encoder2 = nn.Sequential(
            Encoder(c_1, c_2, 3, 2),
            Encoder(c_2, c_2, 3, 1),
        )
        self.encoder3 = nn.Sequential(
            Encoder(c_2, c_3, 3, 2),
            Encoder(c_3, c_3, 3, 1),
        )
        self.encoder4 = nn.Sequential(
            Encoder(c_3, c_4, 3, 2),
            Encoder(c_4, c_4, 3, 1),
        )
        self.encoder5 = nn.Sequential(
            Encoder(c_4, c_5, 3, 2),
            Encoder(c_5, c_5, 3, 1),
        )
        self.encoder6 = nn.Sequential(
            Encoder(c_5, c_6, 3, 2),
            Encoder(c_6, c_6, 3, 1),
        )


        RIRB0_0 = [RIRB(c_6, 10) for _ in range(4)]
        RIRB0_0.append(nn.Conv2d(c_6, c_6, 3, 1, 1))

        RIRB0_1 = [RIRB(c_6, 10) for _ in range(4)]
        RIRB0_1.append(nn.Conv2d(c_6, c_6, 3, 1, 1))

        RIRB0_2 = [RIRB(c_6, 10) for _ in range(4)]
        RIRB0_2.append(nn.Conv2d(c_6, c_6, 3, 1, 1))

        self.RIRB0 = nn.ModuleList([
            nn.Sequential(*RIRB0_0), 
            nn.Sequential(*RIRB0_1), 
            nn.Sequential(*RIRB0_2)
        ])

        self.decoder6 = nn.ModuleList([
            Decoder(c_6, c_5, 3, 1),
            Decoder(c_6, c_5, 3, 1), 
            Decoder(c_6, c_5, 3, 1),
        ])
        self.decoder5 = nn.ModuleList([
            Decoder((n_out+1)*c_5, c_4, 3, 1),
            Decoder((n_out+1)*c_5, c_4, 3, 1),
            Decoder((n_out+1)*c_5, c_4, 3, 1),
        ])
        self.decoder4 = nn.ModuleList([
            Decoder((n_out+1)*c_4, c_3, 3, 1),
            Decoder((n_out+1)*c_4, c_3, 3, 1),
            Decoder((n_out+1)*c_4, c_3, 3, 1),
        ])
        self.decoder3 = nn.ModuleList([
            Decoder((n_out+1)*c_3, c_2, 3, 1),
            Decoder((n_out+1)*c_3, c_2, 3, 1),
            Decoder((n_out+1)*c_3, c_2, 3, 1),
        ])
        self.decoder2 = nn.ModuleList([
            Decoder((n_out+1)*c_2+c_out_num, c_1, 3, 1),
            Decoder((n_out+1)*c_2+c_out_num, c_1, 3, 1),
            Decoder((n_out+1)*c_2+c_out_num, c_1, 3, 1),
        ])
        self.decoder1 = nn.ModuleList([
            Decoder((n_out+1)*c_1+c_out_num, c_0, 3, 1),
            Decoder((n_out+1)*c_1+c_out_num, c_0, 3, 1),
            Decoder((n_out+1)*c_1+c_out_num, c_0, 3, 1),
        ])

        self.create_output4 = CreateOutput((n_out+1)*c_3, w, 4)
        self.create_output3 = CreateOutput((n_out+1)*c_2+c_out_num, w, 3)
        self.create_output2 = CreateOutput((n_out+1)*c_1+c_out_num, w, 2)
        self.create_output1 = CreateOutput((n_out+1)*c_0+c_out_num, w, 1)

        self.normalize_output4 = NormalizeOutput(w, 4)
        self.normalize_output3 = NormalizeOutput(w, 3)
        self.normalize_output2 = NormalizeOutput(w, 2)

    def forward(self, x: Tensor) -> List[List[Tensor]]:
        opt = self.opt
        n_out = 3

        # upsampling
        x = nn.functional.interpolate(x, (512,512), mode='bicubic', align_corners=True)

        # encoder
        conv0 = self.encoder0(x)
        conv1 = self.encoder1(conv0)
        conv2 = self.encoder2(conv1)
        conv3 = self.encoder3(conv2)
        conv4 = self.encoder4(conv3)
        conv5 = self.encoder5(conv4)
        conv6 = self.encoder6(conv5)

        # decoder
        deconv6 = []
        deconv5 = []
        deconv4 = []
        deconv3 = []
        deconv2 = []
        deconv1 = []
        results = []

        for i in range(n_out):
            deconv6.append(self.decoder6[i](conv6 + self.RIRB0[i](conv6)))
        deconv6.append(conv5)

        for i in range(n_out):
            deconv5.append(self.decoder5[i](deconv6))
        deconv5.append(conv4)  # deconv5

        for i in range(n_out):
            deconv4.append(self.decoder4[i](deconv5))
        deconv4.append(conv3)  # deconv4

        ms_num = opt.ms_num

        for i in range(n_out):
            deconv3.append(self.decoder3[i](deconv4))
        deconv3.append(conv2)  # deconv3
        
        if ms_num >= 4:
            # scale 4 output
            s4_out = self.create_output4(deconv4)
            s4_out_up = self.normalize_output4(s4_out)
            deconv3.append(s4_out_up)
            results.append(s4_out)

        for i in range(n_out):
            deconv2.append(self.decoder2[i](deconv3))
        deconv2.append(conv1)  # deconv2

        if ms_num >= 3:
            # scale 3 output
            s3_out = self.create_output3(deconv3)
            s3_out_up = self.normalize_output3(s3_out)
            deconv2.append(s3_out_up)
            results.append(s3_out)

        for i in range(n_out):
            deconv1.append(self.decoder1[i](deconv2))
        deconv1.append(conv0)  # deconv1

        if ms_num >= 2:
            # scale 2 output
            s2_out = self.create_output2(deconv2)
            s2_out_up = self.normalize_output2(s2_out)
            deconv1.append(s2_out_up)
            results.append(s2_out)

        s1_out = self.create_output1(deconv1)
        results.append(s1_out)

        return results
