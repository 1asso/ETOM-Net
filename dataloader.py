import torch.multiprocessing as mp
import torch
import math
import easydict as edict
import os


def create(opt):
    dataset = ETOMDataset(opt, split)
    loaders_0 = DataLoader(dataset, opt, 'train')
    loaders_1 = DataLoader(dataset, opt, 'val')
    return loaders_0, loaders_1

class ETOMDataset:
    def __init__(self, opt, split):
        self.opt = opt
        self.split = split
        if split == 'train':
            self.image_list = os.path.join(opt.data_dir, opt.train_list)
            self.dir = os.path.join(opt.data_dir, 'train/Images')
        elif split = 'val':
            self.image_list = os.path.join(opt.data_dir, opt.val_list)
            self.dir = os.path.join(opt.data_dir, 'val/Images')
        
        self.image_info = self.read_list(self.image_list)

        print('dataset filenames: {}'.format(self.image_list))
        print('dataset image directory: {}'.format(self.dir))
        print('image size H * W: {} * {}'.format(self.opt.scale_h, self.opt.scale_w))
        print()

    def read_list(file_path):
        image_info = {}
        ct = 0
        f = open(file_path, 'r')
        for line in f.readlines():
            image_info[ct] = line 
            ct += 1

        print('totaling {} images in split {}'.format(ct, self.split))
        return image_info
    
    def get(i):
        

class DataLoader:
    def process(self, indices):
        sz = len(indices)
        batch = masks = rhos = flows = trimaps = image_size = None
        for i, idx in enumerate(indices.tolist()):
            sample = self.dataset.get(idx)
            input = sample.input
            if not batch:
                image_size = input.size()
                batch = torch.FloatTensor(sz, *image_size)
                masks = torch.FloatTensor(sz, image_size[1], image_size[2]) 
                rhos = torch.FloatTensor(sz, image_size[1], image_size[2]) 
                flows = torch.FloatTensor(sz, 3, image_size[1], image_size[2]) 
                if self.in_trimap:
                    trimaps = torch.FloatTensor(sz, image_size[1], image_size[2]) 
            
            batch[i] = input.clone()
            masks[i] = sample.mask.clone()
            rhos[i] = sample.rho.clone()
            flows[i] = sample.flow.clone()
            if self.in_trimap:
                trimaps[i] = sample.trimap.clone()
            
        batch_sample = edict()
        batch_sample.input = batch
        batch_sample.masks = masks
        batch_sample.rhos = rhos
        batch_sample.flows = flows

        if self.in_trimap:
            batch_sample.trimaps = trimaps
            
        return batch_sample

    def __init__(self, dataset, opt, split):
        self.opt = opt
        self.dataset = dataset
        self.split = split

        manual_seed = opt.manual_seed
        if manual_seed != 0:
            torch.manual_seed(manual_seed + idx)
        # torch.set_num_threads(1)

        self.size = dataset.size()
        self.batch_size = opt.batch_size
        self.in_trimap = opt.in_trimap

    def get_num_of_batches(self):
        return math.ceil(self.size / self.batch_size)

    def get_size(self):
        return self.size

    def run(self, max_num):
        processes = self.opt.processes
        size = (man_num and max_num > 0) and min(max_num, self.size) or self.size
        perm = (self.split == 'val') and torch.arange(size) or torch.randperm(size)
        print('[dataloader run on split: {} with size: {} / {}'.format(self.split, size, self.size))

        idx = 0
        results = []

        print('Creating pool with {} processes\n'.format(processes))

        with mp.Pool(processes) as pool:
            task = []
            while idx < size:
                indices = torch.narrow(perm, 0, idx, min(self.batch_size, size-idx))
                tasks.append(indices)
                idx += self.batch_size
                
            for i in range(len(tasks)):
                results.append((i, pool.apply_async(self.process, tasks[i])))
            
            return results





    


