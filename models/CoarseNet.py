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
        # All tensors in the input list must either have the same shape
        # (except in the concatenating dimension) or be empty.
        # input: [4 * Tensor(batch_size, 128, feature_map(h, w))]

        x = torch.cat(x, dim=1) 
        # Use dim=1 because of the existence of batch_size
        # output: Tensor(batch_size, 4 * 128, feature_map(h, w))
        
        flow = self.conv1_0(x).cuda()
        flow = self.th(flow).clone()
        flow *= self.ratio


        mask = self.conv1_1(x).cuda()

        rho = self.conv2(x).cuda()
        #if input.size()[3] == 128: 
        #    print(flow[1,:,1,1])
        #    print(mask[1,:,1,1])
        #    print(rho[1,:,1,1],end='\n---\n')

        return [flow, mask, rho]


class NormalizeOutput(nn.Module):
    def __init__(self, w: int, scale: int) -> None:
        super(NormalizeOutput, self).__init__()
        self.ratio = 2.0 ** (scale-1) / w
        self.up = nn.Upsample(scale_factor=2)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x: List[Tensor]) -> Tensor:
        # input should be a list: [flow, mask, rho]

        x[0] *= self.ratio
        x[1] = self.sm(x[1])

        x = torch.cat(x, dim=1)
        return self.up(x)
        

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, step: int, use_BN: bool) -> None:
        super(Encoder, self).__init__()
        pad = math.floor((k-1)/2)
        od = OrderedDict()
        od['conv'] = nn.Conv2d(in_channels, out_channels, k, step, pad)
        if use_BN:
            od['batch_norm'] = nn.BatchNorm2d(out_channels)
        
        od['actv'] = nn.ReLU(inplace=True)
        
        self.encoder = nn.Sequential(od).cuda()

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, step: int, is_bottom: bool, use_BN: bool) -> None:
        super(Decoder, self).__init__()
        pad = math.floor((k-1)/2)
        self.is_bottom = is_bottom
        od = OrderedDict()
        od['conv'] = nn.Conv2d(in_channels, out_channels, k, step, pad)
        if use_BN:
            od['batch_norm'] = nn.BatchNorm2d(out_channels)
        
        od['actv'] = nn.ReLU(inplace=True)
        od['up'] = nn.Upsample(scale_factor=2)
        
        self.decoder = nn.Sequential(od).cuda()

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Tensor:
        if not self.is_bottom:
            x = torch.cat(x, dim=1)

        return self.decoder(x)

class CoarseNet(nn.Module):
    def __init__(self, opt: Namespace) -> None:
        super(CoarseNet, self).__init__()
        self.opt = opt
        use_BN = opt.use_BN
        w = opt.crop_w
        c_in = 3
        if opt.in_trimap:
            c_in += 1
        if opt.in_bg:
            c_in += 3

        c_0 = c_1 = 16
        c_2 = 32
        c_3 = 64
        c_4 = 128
        c_5 = c_6 = 256

        self.encoder0 = nn.Sequential(
            Encoder(c_in, c_0, 3, 1, use_BN),
            Encoder(c_0, c_0, 3, 1, use_BN)
        ).cuda()
        self.encoder1 = nn.Sequential(
            Encoder(c_0, c_1, 3, 2, use_BN),
            Encoder(c_1, c_1, 3, 1, use_BN)
        ).cuda()
        self.encoder2 = nn.Sequential(
            Encoder(c_1, c_2, 3, 2, use_BN),
            Encoder(c_2, c_2, 3, 1, use_BN)
        ).cuda()
        self.encoder3 = nn.Sequential(
            Encoder(c_2, c_3, 3, 2, use_BN),
            Encoder(c_3, c_3, 3, 1, use_BN)
        ).cuda()
        self.encoder4 = nn.Sequential(
            Encoder(c_3, c_4, 3, 2, use_BN),
            Encoder(c_4, c_4, 3, 1, use_BN)
        ).cuda()
        self.encoder5 = nn.Sequential(
            Encoder(c_4, c_5, 3, 2, use_BN),
            Encoder(c_5, c_5, 3, 1, use_BN)
        ).cuda()
        self.encoder6 = nn.Sequential(
            Encoder(c_5, c_6, 3, 2, use_BN),
            Encoder(c_6, c_6, 3, 1, use_BN)
        ).cuda()

        c_0 = c_1 = 16
        c_2 = 32
        c_3 = 64
        c_4 = 128
        c_5 = c_6 = 256

        n_out = 3  # num of output branches (flow, mask, rho)
        c_out_num = 5  # total num of channels (2 + 2 + 1)

        self.decoder6 = nn.ModuleList([
            Decoder(c_6, c_5, 3, 1, True, use_BN).cuda(),
            Decoder(c_6, c_5, 3, 1, True, use_BN).cuda(), 
            Decoder(c_6, c_5, 3, 1, True, use_BN).cuda()
        ])
        self.decoder5 = nn.ModuleList([
            Decoder((n_out+1)*c_5, c_4, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_5, c_4, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_5, c_4, 3, 1, False, use_BN).cuda()
        ])
        self.decoder4 = nn.ModuleList([
            Decoder((n_out+1)*c_4, c_3, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_4, c_3, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_4, c_3, 3, 1, False, use_BN).cuda()
        ])
        self.decoder3 = nn.ModuleList([
            Decoder((n_out+1)*c_3, c_2, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_3, c_2, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_3, c_2, 3, 1, False, use_BN).cuda()
        ])
        self.decoder2 = nn.ModuleList([
            Decoder((n_out+1)*c_2+c_out_num, c_1, 3, 1, False,use_BN).cuda(),
            Decoder((n_out+1)*c_2+c_out_num, c_1, 3, 1, False,use_BN).cuda(),
            Decoder((n_out+1)*c_2+c_out_num, c_1, 3, 1, False,use_BN).cuda()
        ])
        self.decoder1 = nn.ModuleList([
            Decoder((n_out+1)*c_1+c_out_num, c_0, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_1+c_out_num, c_0, 3, 1, False, use_BN).cuda(),
            Decoder((n_out+1)*c_1+c_out_num, c_0, 3, 1, False, use_BN).cuda()
        ])

        self.create_output4 = CreateOutput((n_out+1)*c_3, w, 4).cuda()
        self.create_output3 = CreateOutput((n_out+1)*c_2+c_out_num, w, 3).cuda()
        self.create_output2 = CreateOutput((n_out+1)*c_1+c_out_num, w, 2).cuda()
        self.create_output1 = CreateOutput((n_out+1)*c_0+c_out_num, w, 1).cuda()

        self.normalize_output4 = NormalizeOutput(w, 4).cuda()
        self.normalize_output3 = NormalizeOutput(w, 3).cuda()
        self.normalize_output2 = NormalizeOutput(w, 2).cuda()


    def forward(self, x: Tensor) -> List[List[Tensor]]:
        
        
        opt = self.opt
        n_out = 3

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
            deconv6.append(self.decoder6[i](conv6))
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