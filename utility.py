from models.CoarseNet import CreateOutput
import os
import torch
from torch import Tensor
import math
import logging
from torchvision import transforms
from torchvision.utils import save_image
import struct
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from skimage.color import hsv2rgb
from typing import Type, Any, Callable, Union, List, Optional, Dict

TAG = 202021.25


### IO utilities


def load_data(f_name: str) -> "model":
    data = {}
    if os.path.isfile(f_name):
        data = torch.load(f_name)

    return data


def resize_tensor(input_tensor: Tensor, h: int, w: int) -> Tensor:
    final_output = None

    for img in input_tensor:
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    return final_output


def save_compact_results(save_name: str, results: List[Tensor], width_num: int) -> None:
    _int = 5
    num = len(results)
    w_n = width_num or 3
    h_n = math.ceil(num / w_n)
    idx = 1
    big_img = None
    fix_h = fix_w = None
    h = w = None
    
    for v in results:
        if not type(v) == bool:
            img = v.float()
            if img.dim() > 3 or img.dim() < 2:
                logging.error('Dim of image must be 2 or 3')
            if big_img == None:
                c, h, w = list(img.size())
                fix_h = h
                fix_w = w
                big_img = torch.Tensor(3, h_n*h + (h_n-1)*_int,
                                        w_n*w + (w_n-1)*_int).fill_(0)
            if img.size(0) != 3:
                img = img.unsqueeze(0)
                img = img.repeat(3, 1, 1)
            if img.size(1) != fix_h or img.size(2) != fix_w:
                img = resize_tensor(img, fix_h, fix_w)

            h_idx = math.floor((idx-1) / w_n) + 1
            w_idx = (idx-1) % w_n + 1
            h_start = (h_idx-1) * (h+_int)
            w_start = (w_idx-1) * (w+_int)
            big_img[:, h_start:h_start+h, w_start:w_start+w] = img
        idx += 1
    path = Path(save_name)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    save_image(big_img, save_name)


### flow utilities


def flow_to_color(flow: Tensor) -> Tensor:
    flow = flow.float()
    if flow.size(0) == 3:
        f_val = flow[2, :, :].ge(0.1).float()
    else:
        f_val = torch.ones(flow.size(1), flow.size(2)).cuda()
    
    f_du = flow[1, :, :].clone()
    f_dv = flow[0, :, :].clone()

    f_mag = torch.sqrt(torch.pow(f_du, 2) + torch.pow(f_dv, 2))
    f_dir = torch.atan2(f_dv, f_du)
    img = flow_mapping(f_mag, f_dir, f_val)
    
    return img


def flow_mapping(f_mag: Tensor, f_dir: Tensor, f_val: Tensor) -> Tensor:
    img_size = f_mag.size()
    img = torch.zeros(3, img_size[0], img_size[1]).cuda()

    img[0, :, :] = (f_dir + math.pi) / (2 * math.pi)
    img[1, :, :] = torch.div(f_mag, (f_mag.size(1) * 0.5)).clamp(0, 1)
    img[2, :, :] = 1

    img[1:2, :, :] = torch.minimum(torch.maximum(img[1:2, :, :], torch.zeros(img_size).cuda()), torch.ones(img_size).cuda())
    
    img = torch.from_numpy(hsv2rgb(img.cpu().permute(1,2,0).detach())).cuda().permute(2,0,1)

    img[0, :, :] = img[0, :, :] * f_val
    img[1, :, :] = img[1, :, :] * f_val
    img[2, :, :] = img[2, :, :] * f_val

    return img


def load_flow(filename: str) -> Tensor:
    f = open(filename, 'rb')
    tag = struct.unpack('f', f.read(4))[0]
    assert tag == TAG, 'Unable to read ' + filename + ' because of wrong tag'

    w = struct.unpack('i', f.read(4))[0]
    h = struct.unpack('i', f.read(4))[0]
    channels = 2

    l = [] # in file: [h, w, c]
    for i, val in enumerate(struct.iter_unpack('h', f.read())):
        if not i % 2:
            l.append([])
        l[int(i/2)].append(val[0])

    flow = torch.ShortTensor(l).reshape(h, w, channels)
    f.close()

    flow = flow.permute(2, 0, 1).float() # output: [c, h, w]
    return flow


def save_flow(filename: str, flow: Tensor) -> None:
    flow = flow.short().permute(1, 2, 0).clone()
    f = open(filename, 'wb')
    f.write(struct.pack('f', TAG))
    f.write(struct.pack('i', flow.size(1)))
    f.write(struct.pack('i', flow.size(0)))
    for val in flow.reshape([flow.numel()]).tolist():
        f.write(struct.pack('h', val))

    f.close()


### dict utilities


def build_loss_string(losses: dict) -> str:
    total_loss = 0
    s = ''
    count = 0
    for k, v in losses.items():
        count += 1
        s += f'{k}: {v}'
        if not count % 4:
            s += '\n'
        elif count != 16:
            s += ', '
        total_loss += v

    s += f'[Total Loss: {total_loss}]'
    return s


def dicts_add(dict_ori: dict, dict_to_add: dict) -> None:
    for k, v in dict_to_add.items():
        if not k in dict_ori.keys():
            dict_ori[k] = 0
        dict_ori[k] = dict_ori[k] + v


def dict_of_dict_average(dict_of_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    result = {}
    for k1, v1 in dict_of_dict.items():
        for k2, v2 in v1.items():
            if not k2 in result.keys():
                result[k2] = 0
            result[k2] = result[k2] + v2
    n = len(dict_of_dict)
    for k, v in result.items():
        result[k] /= n
    return result


def dict_divide(dict_ori: dict, n: int) -> dict:
    return {k: v / n for k, v in dict_ori.items()}


def hist_to_str(d: dict):
    s = ''
    for k, v in d.items():
        s += f'Epoch: {k}\n{v}\n\n'
    return s


### model utilities


class CreateMultiScaleData(nn.Module):
    def __init__(self, ms_num: int) -> None:
        super(CreateMultiScaleData, self).__init__()
        self.ms_num = ms_num

    def forward(self, x: List[Tensor]) -> List[List[Tensor]]:
        result = [[],[],[],[],[]]
        for i in range(self.ms_num, 0, -1):
            scale = 2**(i-1)
            result[0].append(nn.AvgPool2d((scale, scale))(x[0]))
            result[1].append(nn.AvgPool2d((scale, scale))(x[1]))
            result[2].append(nn.AvgPool2d((scale, scale))(x[2]))
            result[3].append(nn.MaxPool2d((scale, scale))(x[3]))
            result[4].append(nn.AvgPool2d((scale, scale))(x[4]).mul(1/scale))

        return result


class CreateMultiScaleWarping(nn.Module):
    def __init__(self, ms_num: int) -> None:
        super(CreateMultiScaleWarping, self).__init__()
        self.ms_num = ms_num

    def forward(self, x: List[List[Tensor]]) -> List[Tensor]:
        warping_module = []
        for i in range(self.ms_num):
            input_0 = x[0][i] # multi_ref_images
            input_1 = x[1][i] # flows

            single_warping = create_single_warping([input_0, input_1])
            warping_module.append(single_warping)

        return warping_module


def create_single_warping(input: List[Tensor]) -> Tensor:
    ref = input[0]
    flo = input[1]
    grid = grid_generator(flo)

    output = F.grid_sample(ref, grid, align_corners=True)
    return output


def grid_generator(flow: Tensor) -> Tensor:
    B, C, H, W = flow.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float().cuda()
    
    flow = flow.div(H/2)
    flow_clo = flow.clone()
    flow[:,0,:,:] = flow_clo[:,1,:,:]
    flow[:,1,:,:] = flow_clo[:,0,:,:]
    
    # scale grid to [-1,1] 
    grid[:,0,:,:] = 2.0*grid[:,0,:,:].clone() / max(W-1,1)-1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:].clone() / max(H-1,1)-1.0
    
    grid = grid + flow
    
    grid = grid.permute(0,2,3,1)
    return grid


### evaluation utilities


class EPELoss(nn.Module):
    def __init__(self) -> None:
        super(EPELoss, self).__init__()

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor, rho: Tensor) -> Tensor:
        target = target.narrow(1, 0, 2)
        mask = mask.expand_as(target)
        pred = pred * mask * rho
        target = target * mask * rho

        return torch.norm(target-pred, dim=1).mean()


def get_final_pred(ref_img: Tensor, pred_img: Tensor, pred_mask: Tensor, pred_rho: Tensor) -> Tensor:
    final_pred_img = torch.mul(1 - pred_mask, ref_img) + torch.mul(pred_mask, torch.mul(pred_img, pred_rho))
    return final_pred_img


def get_mask(masks: Tensor) -> Tensor:
    n, c, h, w = list(masks.size())
    m = masks.transpose(1, 3).transpose(1,2)
    m = m.reshape(int(m.numel()/m.size(3)), m.size(3))
    _, pred = m.max(1)
    pred = pred.reshape(n, 1, h, w)
    return pred
