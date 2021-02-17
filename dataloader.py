import torch.multiprocessing as mp
import torch
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
import matplotlib.pyplot as plt


def create(opt):
    dataset = ETOMDataset(opt, 'train')
    loaders_0 = DataLoader(dataset, opt, 'train')
    dataset = ETOMDataset(opt, 'val')
    loaders_1 = DataLoader(dataset, opt, 'val')
    return loaders_0, loaders_1

class ETOMDataset:
    def __init__(self, opt, split):
        self.opt = opt
        self.split = split
        if split == 'train':
            self.image_list = os.path.join(opt.data_dir, opt.train_list)
            self.dir = os.path.join(opt.data_dir, 'train/images')
        elif split == 'val':
            self.image_list = os.path.join(opt.data_dir, opt.val_list)
            self.dir = os.path.join(opt.data_dir, 'val/images')
        
        self.image_info = pd.read_csv(self.image_list, sep='\n', header=None)
        print('totaling {} images in split {}'.format(len(self.image_info), self.split))

        print('dataset filenames: {}'.format(self.image_list))
        print('dataset image directory: {}'.format(self.dir))
        print('image size H * W: {} * {}'.format(self.opt.scale_h, self.opt.scale_w))
        print()

    def transform(self, image):
        image = image
        return image

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        sc_w = self.opt.scale_w
        sc_h = self.opt.scale_h
        cr_w = self.opt.crop_w
        cr_h = self.opt.crop_h

        if self.opt.data_aug and self.split == 'train':
            # randomize rescale size
            sc_w = int(random.uniform(cr_w, sc_w * 1.05))
            sc_h = int(random.uniform(cr_h, sc_h * 1.05))

        path = self.image_info.iloc[idx, 0]
        path_base = path.split('_')[0]
        path_ref = path_base + '_ref_2x.png'  # TODO
        path_tar = path_base + '_img_2x.png'  # TODO
        path_mask = path_base + '_mask_2x.png'
        path_rho = path_base + '_rho_2x.png'
        path_flow = path_base + '_flow_2x.flo'

        image_ref = Image.open(os.path.join(self.dir, path_ref))
        image_ref = TF.to_tensor(image_ref) # size: [3, h, w]

        image_tar = Image.open(os.path.join(self.dir, path_tar)) 
        image_tar = TF.to_tensor(image_tar) # size: [3, h, w]

        mask = Image.open(os.path.join(self.dir, path_mask))
        mask = TF.to_tensor(mask.convert('L')) # size: [1, h, w]
        mask.apply_(lambda x: 1 if x else 0)

        rho = Image.open(os.path.join(self.dir, path_rho))
        rho = TF.to_tensor(rho.convert('L')) # size: [1, h, w]

        flow = utility.load_short_flow_file(os.path.join(self.dir, path_flow)) # size: [2, h, w]
        add_on= torch.ones(1, flow.size(1), flow.size(2)) # size: [1, h, w]
        flow = torch.cat([flow, add_on], 0) # size: [3, h, w]

        print(path_base)
        # torch.set_printoptions(profile="full")
        # with open('sample.txt', 'w') as f:
        #     f.write(str(flow))
        # torch.set_printoptions(profile="default")

        # plt.imshow(image_tar.permute(1, 2, 0))
        # plt.imshow(mask.permute(1, 2, 0), cmap='gray')
        # plt.imshow(rho.permute(1, 2, 0), cmap='gray')
   
        # plt.imshow(flow.permute(1, 2, 0))
        # plt.show()


        # check if rescaling or croping is needed
        _, in_h, in_w = image_ref.size()
        need_scale = in_h != sc_h or in_w != sc_w
        need_aug = self.opt.data_aug and (self.split == 'train' or self.split == 'val')
        need_flip = need_aug and random.uniform(0, 1) > 0.5
        need_rotate = need_aug and self.opt.rot_ang and self.split == 'train'
        need_crop = (sc_h != cr_h or sc_w != cr_w) and self.split == 'train'

        if need_scale:
            pass
            # size = [sc_h, sc_w]
            # image_ref = F.interpolate(image_ref, size, mode='bicubic')
            # image_tar = F.interpolate(image_tar, size, mode='bicubic')
            # mask = F.interpolate(mask, size, mode='bicubic')
            # rho = F.interpolate(rho, size, mode='bicubic')
            # flow = F.interpolate(flow, size, mode='bicubic')
            # flow[1] *= (sc_w / in_w)
            # flow[0] *= (sc_h / in_h)
        
        if need_aug:
            final = image_tar
            images = torch.cat([image_ref, final], 0)
        #     # increase the intensity of pixels where total internal reflection happens
        #     dark = torch.lt(rho, 0.7)
        #     dark3 = dark.view(1, sc_h, sc_w).expand(3, sc_h, sc_w)
        #     image_tar[dark3] = image_tar[dark3] + Uniform(0.01, 0.2).sample()

        #     # get the regions of boundary and total internal reflection
        #     k_e = torch.rand(1, 1) * 2 + 1
        #     mask_roi = ... 
        else:
            images = torch.cat([image_ref, image_tar], 0)

        if need_flip:
            pass

        if need_rotate:
            pass

        if need_crop:
            pass

        sample = {}
        sample['input'] = images
        sample['mask'] = mask + 1
        sample['rho'] = rho
        sample['flow'] = flow

        return sample


class DataLoader:
    def process(self, indices):
        sz = len(indices)
        batch = masks = rhos = flows = trimaps = image_size = None
        for i, idx in enumerate(indices.tolist()):
            sample = self.dataset[idx]

            input = sample['input']
            if batch is None:
                image_size = input.size()
                batch = torch.FloatTensor(sz, *image_size)
                masks = torch.FloatTensor(sz, image_size[1], image_size[2]) 
                rhos = torch.FloatTensor(sz, image_size[1], image_size[2]) 
                flows = torch.FloatTensor(sz, 3, image_size[1], image_size[2]) 
                if self.in_trimap:
                    trimaps = torch.FloatTensor(sz, image_size[1], image_size[2]) 
            
            batch[i] = input.clone()
            masks[i] = sample['mask'].clone()
            rhos[i] = sample['rho'].clone()
            flows[i] = sample['flow'].clone()
            if self.in_trimap:
                trimaps[i] = sample['trimap'].clone()
            
        batch_sample = {}
        batch_sample['input'] = batch
        batch_sample['masks'] = masks
        batch_sample['rhos'] = rhos
        batch_sample['flows'] = flows

        if self.in_trimap:
            batch_sample['trimaps'] = trimaps
            
        return batch_sample

    def __init__(self, dataset, opt, split):
        self.opt = opt
        self.dataset = dataset
        self.split = split

        self.manual_seed = opt.manual_seed
        # torch.set_num_threads(1)

        self.size = len(dataset)
        self.batch_size = opt.batch_size
        self.in_trimap = opt.in_trimap

    def get_num_of_batches(self):
        return math.ceil(self.size / self.batch_size)

    def get_size(self):
        return self.size

    def run(self, max_num):
        processes = self.opt.processes
        size = (max_num and max_num > 0) and min(max_num, self.size) or self.size
        perm = (self.split == 'val') and torch.arange(size) or torch.randperm(size)
        print('[dataloader run on split: {} with size: {} / {}'.format(self.split, size, self.size))

        idx = 0
        results = []

        print('Creating pool with {} processes\n'.format(processes))

        with mp.Pool(processes) as pool:
            tasks = []
            while idx < size:
                if self.manual_seed != 0:
                    torch.manual_seed(self.manual_seed + idx)
                indices = torch.narrow(perm, 0, idx, min(self.batch_size, size-idx))
                tasks.append(indices)
                idx += self.batch_size
            
            for i in range(len(tasks)):
                results.append((i, pool.apply(self.process, [tasks[i],])))
            pool.close()
            pool.join()
            
            return results