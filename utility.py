from models.CoarseNet import CreateOutput
import os
import torch
import math
import logging
from torchvision import transforms
from torchvision.utils import save_image
import time
import struct
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

TAG = 202021.25


# IO utilities
def load_t7(condition, f_name):
    t7_file = None
    if condition:
        if os.path.isfile(f_name):
            t7_file = torch.load(f_name)

    return t7_file

def resize_tensor(input_tensors, h, w):
    final_output = None

    for img in input_tensors:
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    return final_output

def plot_results_compact(results, log_dir, split):
    pass

def save_results_compact(save_name, results, width_num):
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
            # print(img.size(), big_img.size())
            big_img[:, h_start:h_start+h, w_start:w_start+w] = img
        idx += 1
    path = Path(save_name)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    save_image(big_img, save_name)

# str utilities

def build_loss_string(losses):
    total_loss = 0
    s = ''
    for k, v in losses.items():
        s += '{}: {}, '.format(k, v)
        total_loss += v

    s += ' [Total Loss: {}]'.format(total_loss) 
    return s

# flow utilities

def flow_to_color(flow):
    flow = flow.float()
    if flow.size(0) == 3:
        f_val = flow[2, :, :].ge(0.1).float()
    else:
        f_val = torch.ones(flow.size(1), flow.size(2)).cuda()
    
    f_du = flow[1, :, :].clone()
    f_dv = flow[0, :, :].clone()
    u_max = torch.max(torch.abs(f_du.masked_select(f_val.bool())))
    v_max = torch.max(torch.abs(f_dv.masked_select(f_val.bool())))
    f_max = torch.max(u_max, v_max)

    f_mag = torch.sqrt(torch.pow(f_du, 2) + torch.pow(f_dv, 2))
    f_dir = torch.atan2(f_dv, f_du)
    img = flow_map(f_mag, f_dir, f_val, f_max, 8)
    
    return img

def flow_map(f_mag, f_dir, f_val, f_max, n):
    img_size = f_mag.size()
    img = torch.zeros(3, img_size[0], img_size[1]).cuda()

    img[0, :, :] = (f_dir + math.pi) / (2 * math.pi)
    img[1, :, :] = torch.div(f_mag, (f_mag.size(1) * 0.5)).clamp(0, 1)
    img[2, :, :] = 1

    img[1:2, :, :] = torch.minimum(torch.maximum(img[1:2, :, :], torch.zeros(img_size).cuda()), torch.ones(img_size).cuda())

    img[0, :, :] = img[0, :, :] * f_val
    img[1, :, :] = img[1, :, :] * f_val
    img[2, :, :] = img[2, :, :] * f_val
    
    return img


def load_short_flow_file(filename):
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

def save_short_flow_file(filename, flow):
    flow = flow.short().permute(1, 2, 0).clone()
    f = open(filename, 'wb')
    f.write(struct.pack('f', TAG))
    f.write(struct.pack('i', flow.size(1)))
    f.write(struct.pack('i', flow.size(0)))
    for val in flow.reshape([flow.numel()]).tolist():
        f.write(struct.pack('h', val))

    f.close()

# dict utilities

def dicts_add(dict, dict_to_add):
    for k, v in dict_to_add.items():
        if not k in dict.keys():
            dict[k] = 0
        dict[k] = dict[k] + v

def insert_sub_dict(_dict, sub_dict):
    if _dict:
        _dict.update(sub_dict)
    else:
        _dict = sub_dict.copy()

def dict_of_dict_average(dict_of_dict):
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

def dict_divide(t, n):
    return {k: v / n for k, v in t.items()}

# model utilities

class CreateMultiScaleData(nn.Module):
    def __init__(self, ms_num):
        super(CreateMultiScaleData, self).__init__()
        self.ms_num = ms_num

    def forward(self, input):
        result = [[],[],[],[],[]]
        for i in range(self.ms_num, 0, -1):
            scale = 2**(i-1)
            result[0].append(nn.AvgPool2d((scale, scale))(input[0]))
            result[1].append(nn.AvgPool2d((scale, scale))(input[1]))
            result[2].append(nn.AvgPool2d((scale, scale))(input[2]))
            result[3].append(nn.MaxPool2d((scale, scale))(input[3]))
            result[4].append(nn.AvgPool2d((scale, scale))(input[4]).mul(1/scale))

        return result

class CreateMultiScaleWarping(nn.Module):
    def __init__(self, ms_num):
        super(CreateMultiScaleWarping, self).__init__()
        self.ms_num = ms_num

    def forward(self, input):
        warping_module = []
        for i in range(self.ms_num):
            input_0 = input[0][i] # multi_ref_images
            input_1 = input[1][i] # flows

            single_warping = create_single_warping_module([input_0, input_1])
            warping_module.append(single_warping)

        return warping_module


def create_single_warping_module(_input):
    input = _input[0]
    grid = grid_generator(_input[1])
    output = F.grid_sample(input, grid, align_corners=True)
    return output

def grid_generator(flows):
    batch = flows.size(0)
    height = flows.size(2)
    width = flows.size(3)

    base_grid_extend = torch.Tensor(batch, height, width, 2).cuda()
    base_grid = torch.Tensor(height, width, 2).cuda()

    for i in range(height):
        base_grid[i, :, 0] = -1 + (i-1) / (height-1) * 2

    for i in range(width):
        base_grid[:, i, 1] = -1 + (i-1) / (width-1) * 2

    for i in range(batch):
        base_grid_extend[i, :, :] = base_grid.clone()

    if flows.size(1) == 2:
        flows = flows.transpose(1, 2).transpose(2, 3)
    
    #torch.set_printoptions(profile="full")
    #with open('sample.txt', 'w') as f:
    #    f.write(str(flows))
    #torch.set_printoptions(profile="default")
    #exit()
	
    output = base_grid_extend.add(flows)
    return output


# evaluation utilities

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, pred, target, mask):
        target = target.narrow(1, 0, 2)
        mask = mask.expand_as(target)
        pred = pred * mask
        target = target * mask

        return torch.norm(target-pred, dim=1).mean()

def get_final_pred(ref_img, pred_img, pred_mask, pred_rho):
    final_pred_img = torch.mul(1 - pred_mask, ref_img) + torch.mul(pred_mask, torch.mul(pred_img, pred_rho))
    return final_pred_img

def get_mask(masks):
    n, c, h, w = list(masks.size())
    m = masks.transpose(1, 3).transpose(1,2)
    m = m.reshape(int(m.numel()/m.size(3)), m.size(3))
    _, pred = m.max(1)
    pred = pred.reshape(n, 1, h, w)
    return pred
