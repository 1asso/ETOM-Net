import torch.multiprocessing as mp
import torch
from torch import Tensor
import math
import os
import utility
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import cv2
import numpy as np
from torch.utils.data import DataLoader
from option import args
from argparse import Namespace
from typing import Type, Any, Callable, Union, List, Optional, Tuple, Dict


def create(opt: Namespace) -> Tuple[DataLoader, DataLoader]:
    dataset_0 = ETOMDataset(opt, 'train')
    loader_0 = DataLoader(dataset_0, batch_size=opt.batch_size,
                        shuffle=True, num_workers=8, collate_fn=collate)
    dataset_1 = ETOMDataset(opt, 'val')
    loader_1 = DataLoader(dataset_1, batch_size=opt.batch_size,
                        shuffle=True, num_workers=8, collate_fn=collate)
    return loader_0, loader_1


def collate(sample: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    sz = len(sample)
    batch = masks = rhos = flows = image_size = None
    for i, idx in enumerate(sample):
        s = sample[i]

        images = s['images']
        if batch is None:
            image_size = images.size()
            input = torch.FloatTensor(sz, *s['input'].size())
            batch = torch.FloatTensor(sz, *image_size)
            masks = torch.FloatTensor(sz, image_size[1], image_size[2]) 
            rhos = torch.FloatTensor(sz, image_size[1], image_size[2]) 
            flows = torch.FloatTensor(sz, 3, image_size[1], image_size[2])
        
        input[i] = s['input'].clone()
        batch[i] = images.clone()
        masks[i] = s['mask'].clone()
        rhos[i] = s['rho'].clone()
        flows[i] = s['flow'].clone()
        
    batch_sample = {}
    batch_sample['input'] = input
    batch_sample['images'] = batch
    batch_sample['masks'] = masks
    batch_sample['rhos'] = rhos
    batch_sample['flows'] = flows
    
    return batch_sample


class ETOMDataset(torch.utils.data.Dataset):
    def __init__(self, opt: Namespace, split: str) -> None:
        self.opt = opt
        self.split = split
        if split == 'train':
            self.image_list = os.path.join(opt.data_dir, opt.train_list)
            self.dir = os.path.join(opt.data_dir, 'train/Images')
            if opt.max_train_num > 0:
                self.image_info = pd.read_csv(self.image_list, sep='\n', header=None, nrows=opt.max_train_num)
            else:
                self.image_info = pd.read_csv(self.image_list, sep='\n', header=None)
        elif split == 'val':
            self.image_list = os.path.join(opt.data_dir, opt.val_list)
            self.dir = os.path.join(opt.data_dir, 'val/Images')
            if opt.max_val_num > 0:
                self.image_info = pd.read_csv(self.image_list, sep='\n', header=None, nrows=opt.max_val_num)
            else:
                self.image_info = pd.read_csv(self.image_list, sep='\n', header=None)
        
        print(f'\n\n --> Split: {self.split}')
        print(f'totaling {len(self.image_info)} images')
        print(f'dataset filenames: {self.image_list}')
        print(f'dataset image directory: {self.dir}')
        print(f'image size H * W: {self.opt.scale_h} * {self.opt.scale_w}')

    def transform(self, image: Tensor) -> Tensor:
        image = image
        return image

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sc_w = self.opt.scale_w
        sc_h = self.opt.scale_h
        cr_w = self.opt.crop_w
        cr_h = self.opt.crop_h

        if self.opt.data_aug and self.split == 'train':
            # randomize rescale size
            sc_w = int(random.uniform(cr_w, sc_w * 1.05))
            sc_h = int(random.uniform(cr_h, sc_h * 1.05))

        path = self.image_info.iloc[idx, 0]
        path_base = os.path.splitext(path)[0]
        path_input = path_base + '_1x.jpg'
        path_ref = path_base + '_ref.jpg'
        path_tar = path_base + '.jpg'
        path_mask = path_base + '_mask.png'
        path_rho = path_base + '_rho.png'
        path_flow = path_base + '_flow.flo'

        image_input = Image.open(os.path.join(self.dir, path_input))
        image_input = TF.to_tensor(image_input) # size: [3, h, w]

        image_ref = Image.open(os.path.join(self.dir, path_ref))
        image_ref = TF.to_tensor(image_ref) # size: [3, h, w]

        image_tar = Image.open(os.path.join(self.dir, path_tar)) 
        image_tar = TF.to_tensor(image_tar) # size: [3, h, w]

        mask = Image.open(os.path.join(self.dir, path_mask))
        mask = TF.to_tensor(mask.convert('L')) # size: [1, h, w]
        mask.apply_(lambda x: 1 if x else 0)

        rho = Image.open(os.path.join(self.dir, path_rho))
        rho = TF.to_tensor(rho.convert('L')) # size: [1, h, w]

        flow = utility.load_flow(os.path.join(self.dir, path_flow)) # size: [2, h, w]
        add_on= torch.ones(1, flow.size(1), flow.size(2)) # size: [1, h, w]
        flow = torch.cat([flow, add_on], 0) # size: [3, h, w]

        # torch.set_printoptions(profile="full")
        # with open('sample.txt', 'w') as f:
        #     f.write(str(flow))
        # exit()

        # check if rescaling or croping is needed
        _, in_h, in_w = image_ref.size()
        need_scale = in_h != sc_h or in_w != sc_w
        need_aug = self.opt.data_aug and (self.split == 'train' or self.split == 'val')
        need_flip = need_aug and random.uniform(0, 1) > 0.5
        need_rotate = need_aug and self.opt.rot_ang and self.split == 'train'
        need_crop = (sc_h != cr_h or sc_w != cr_w) and self.split == 'train'

        if need_scale:
            pass
        
        if need_aug:
            final = image_tar
            images = torch.cat([image_ref, final], 0)
        else:
            images = torch.cat([image_ref, image_tar], 0)

        if need_flip:
            pass

        if need_rotate:
            pass

        if need_crop:
            pass

        sample = {}
        sample['input'] = image_input
        sample['images'] = images
        sample['mask'] = mask
        sample['rho'] = rho
        sample['flow'] = flow

        return sample