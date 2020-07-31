import os
import torch
import math
import logging
from PIL import Image
from torchvision import transforms
form torchvision.utils import save_image


class IOUtils:
    def load_t7(condition, f_name):
        if condition:
            if os.path.isfile(f_name):
                t7_file = torch.load(f_name)

        return t7_file

    def resize_tensor(input_tensors, h, w):
        final_output = None
        batch_size, channel, height, width = input_tensors.shape
        input_tensors = torch.squeeze(input_tensors, 1)
        
        for img in input_tensors:
            img_PIL = transforms.ToPILImage()(img)
            img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
            img_PIL = torchvision.transforms.ToTensor()(img_PIL)
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
        for k, v in results.items():
            if v:
                img = v.float()
                if img.dim() > 3 or img.dim() < 2:
                    logging.error('Dim of image must be 2 or 3')
                if not big_img:
                    c, h, w = img.size().tolist()
                    fix_h = h
                    fix_w = w
                    big_img = torch.Tensor(3, h_n*h + (h_n-1)*_int, \
                        w_n*w + (w_n-1)*_int).fill_(0)
                if img.size(0) != 3:
                    img = torch.repeat(img, 3, 1, 1)
                if img.size(1) != fix_h or img.size(2) != fix_w:
                    img = self.resize_tensor(img, fix_h, fix_w)
                
                h_idx = math.floor((idx-1) / w_n) + 1
                w_idx = (idx-1) % w_n + 1
                h_start = 1 + (h_idx-1) * (h+_int)
                w_start = 1 + (w_idx-1) * (w+_int)
                big_img[[[], [h_start:h_start+h-1], [w_start:w_start+w-1]]] = img
            idx += 1
        save_image(big_img, save_image)
        
