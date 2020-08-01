import torch.nn as nn
import torch
import torch.nn.functional as F
import collections.OrderedDict
import math


class CreateOutput(nn.Module):
    def __init__(self, in_channels, w, scale):
        k = 3
        step = 1
        pad = (k - 1) / 2
        self.ratio = w / 2 ** (scale-1)
        self.th = nn.Tanh()
        self.conv1 = nn.Conv2d(in_channels, 2, k, step, pad)
        self.conv2 = nn.Conv2d(in_channels, 1, k, step, pad)


    def forward(self, input):
        # All tensors in the input list must either have the same shape
        # (except in the concatenating dimension) or be empty.
        # input: [4 * Tensor(batch_size, 128, feature_map(w, h))]

        input = torch.cat(input, dim=1) 
        # Use dim=1 because of the existence of batch_size
        # output: Tensor(batch_size, 4 * 128, feature_map(w, h))
        
        flow_path = self.conv1(input)
        flow_path = self.th(flow_path)
        flow_path *= self.ratio

        mask = self.conv1(input)

        rho = self.conv2(input)

        return [flow_path, mask, rho]


class NormalizeOutput(nn.Module):
    def __init__(self, w, scale):
        self.ratio = 2.0 ** (scale-1) / w
        self.up = nn.Upsample(scale_factor=2)
        self.sm = nn.Softmax()

    def forward(self, input):
        # input should be a list: [flow_path, mask, rho]

        input[0] *= self.ratio
        input[1] = self.sm(input[1])

        input = torch.cat(input, dim=1)
        return self.up(input)
        

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, k, step, use_BN):
        pad = math.floor((k-1)/2)
        od = OrderedDict()
        od['conv'] = nn.conv2d(in_channels, out_channels, k, step, pad)
        if use_BN:
            od['batch_norm'] = nn.BatchNorm2d(out_channels)
        
        od['actv'] = nn.ReLU(True)
        
        self.encoder = nn.Sequential(od)

    def forward(self, input):
        return self.encoder(input)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, k, step, is_bottom, use_BN):
        pad = math.floor((k-1)/2)
        self.is_bottom = is_bottom
        od = OrderedDict()
        od['conv'] = nn.conv2d(in_channels, out_channels, k, step, pad)
        if use_BN:
            od['batch_norm'] = nn.BatchNorm2d(out_channels)
        
        od['actv'] = nn.ReLU(True)
        od['up'] = nn.Upsample(scale_factor=2)
        
        self.decoder = nn.Sequential(od)

    def forward(self, input):
        input = torch.cat(input, dim=1)
        return self.decoder(input)

class CoarseNet(nn.Module):
    def __init__(self, config):
        Super(CoarseNet, self).__init__()
        self.w = config.crop_w
        self.h = config.crop_h
        self.use_BN = config.use_BN
        self.c_in = 3
        if config.in_trimap:
            c_in += 1
        if config.in_bg:
            c_in += 3

        c_0 = c_1 = 16
        c_2 = 32
        c_3 = 64
        c_4 = 128
        c_5 = c_6 = 256

        self.encoder0 = nn.Sequential(
            Encoder(c_in, c_0, 3, 1, use_BN),
            Encoder(c_0, c_0, 3, 1, use_BN)
        )
        self.encoder1 = nn.Sequential(
            Encoder(c_0, c_1, 3, 2, use_BN),
            Encoder(c_1, c_1, 3, 1, use_BN)
        )
        self.encoder2 = nn.Sequential(
            Encoder(c_1, c_2, 3, 2, use_BN),
            Encoder(c_2, c_2, 3, 1, use_BN)
        )
        self.encoder3 = nn.Sequential(
            Encoder(c_2, c_3, 3, 2, use_BN),
            Encoder(c_3, c_3, 3, 1, use_BN)
        )
        self.encoder4 = nn.Sequential(
            Encoder(c_3, c_4, 3, 2, use_BN),
            Encoder(c_4, c_4, 3, 1, use_BN)
        )
        self.encoder5 = nn.Sequential(
            Encoder(c_4, c_5, 3, 2, use_BN),
            Encoder(c_5, c_5, 3, 1, use_BN)
        )
        self.encoder6 = nn.Sequential(
            Encoder(c_5, c_6, 3, 2, use_BN),
            Encoder(c_6, c_6, 3, 1, use_BN)
        )
        
    def forward(self, input):
        conv0 = self.encoder0(input)
        conv1 = self.encoder1(conv0)
        conv2 = self.encoder2(conv1)
        conv3 = self.encoder3(conv2)
        conv4 = self.encoder4(conv3)
        conv5 = self.encoder5(conv4)
        conv6 = self.encoder6(conv5)

        deconv6 = []
        deconv5 = []
        deconv4 = []
        deconv3 = []
        deconv2 = []
        deconv1 = []
        outputs = []
        n_out = 3  # num of output branches (flow, mask, rho)
        c_out_num = 5  # total num of channels (2 + 2 + 1)c

        for i in range(n_out):
            deconv6[i] = Decoder(c_6, c_5, 3, 1, True, use_BN)(conv6)
        deconv6[n_out] = conv5

        for i in range(n_out):
            deconv5[i] = Decoder((n_out+1) * c_5, c_4, 3, 1, False, use_BN)(conv5)
        deconv5[n_out] = conv4  # deconv5: 24 * 32

        for i in range(n_out):
            deconv4[i] = Decoder((n_out+1) * c_4, c_3, 3, 1, False, use_BN)(conv4)
        deconv4[n_out] = conv3  # deconv4: 48 * 64

        idx = 1
        c_out = 0

        for i in range(n_out):
            deconv3[i] = Decoder((n_out+1) * c_3 + c_out, c_2, 3, 1, False, use_BN)(conv3)
        deconv3[n_out] = conv2  # deconv3: 96 * 128

        for i in range(n_out):
            deconv2[i] = Decoder((n_out+1) * c_2 + c_out, c_1, 3, 1, False, use_BN)(conv2)
        deconv2[n_out] = conv1  # deconv2: 192 * 256

        for i in range(n_out):
            deconv1[i] = Decoder((n_out+1) * c_1 + c_out, c_0, 3, 1, False, use_BN)(conv1)
        deconv1[n_out] = conv0  # deconv1: 384 * 512