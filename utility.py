import os
import torch
import math
import logging
from torchvision import transforms
from torchvision.utils import save_image
import time
import struct

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
    input_tensors = torch.squeeze(input_tensors, 1)

    for img in input_tensors:
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1)
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
    for _, v in results.items():
        if v:
            img = v.float()
            if img.dim() > 3 or img.dim() < 2:
                logging.error('Dim of image must be 2 or 3')
            if not big_img:
                c, h, w = img.size().tolist()
                fix_h = h
                fix_w = w
                big_img = torch.Tensor(3, h_n*h + (h_n-1)*_int,
                                       w_n*w + (w_n-1)*_int).fill_(0)
            if img.size(0) != 3:
                img = torch.repeat(img, 3, 1, 1)
            if img.size(1) != fix_h or img.size(2) != fix_w:
                img = resize_tensor(img, fix_h, fix_w)

            h_idx = math.floor((idx-1) / w_n) + 1
            w_idx = (idx-1) % w_n + 1
            h_start = 1 + (h_idx-1) * (h+_int)
            w_start = 1 + (w_idx-1) * (w+_int)
            big_img[:, h_start:h_start+h-1, w_start:w_start+w-1] = img
        idx += 1
    save_image(big_img, save_image)

# str utilities

def build_loss_string(losses, no_total):
    total_loss = 0
    s = ''
    for k, v in losses.items():
        s += '{}: {}, '.format(k, v)
        total_loss += v
    if not no_total:
        s += ' [Total Loss: {}]'.format(total_loss) 
    return s

def time_left(start_time, n_epochs, batches, cur_epoch, cur_iter):
    cur_time = time.time()
    time_so_far = (cur_time - start_time) / 3600.0
    total_step = n_epochs * batches + cur_iter
    cur_step = (cur_epoch - 1) * batches + cur_iter
    time_left = time_so_far * (total_step / cur_step - 1)
    s = 'Time elapsed: {} hours | Time left: {} hours'.format(time_so_far, time_left)
    return s

def build_time_string(times, no_total):
    total_time = 0
    s = ''
    for k, v in times.items():
        s += '{}: {} seconds, '.format(k, v)
        total_time += v
    if not no_total:
        s += ' [Total time: {}'.format(total_time)
    return s

# flow utilities

def flow_to_color(flow):
    pass

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

def insert_sub_dict(_dict, sub_dict):
    _dict.update(sub_dict)

def dict_of_dict_average(dict_of_dict):
    result = {}
    for k1, v1 in dict_of_dict.items():
        for k2, v2 in v1.items():
            if result[k2] == None:
                result[k2] = 0
            result[k2] = result[k2] + v2
    n = len(dict_of_dict)
    for k, v in result.items():
        result[k] /= n

def dict_divide(t, n):
    return {k: v / n for k, v in t.iteritems()}

# model utilities

def create_multi_scale_data(opt):
    pass

def create_single_warping_module():
    pass

# evaluation utilities

def get_final_pred(ref_img, pred_img, pred_mask, pred_rho):
    pass

def get_mask(masks, repeat3):
    pass

def cal_IoU_mask(gt_mask, pred_mask):
    pass

def cal_err_rho(gt_rho, pred_rho, roi, gt_mask):
    pass
