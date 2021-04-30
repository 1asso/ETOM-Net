import os
import torch
import torch.nn as nn
import math
from argparse import Namespace
from typing import Tuple
from models import CoarseNet
import utility


def update_history(opt: Namespace, epoch: int, loss: float, split: str) -> "model":

    train_hist = utility.load_data(os.path.join(opt.save, 'train_hist.pt'))
    val_hist = utility.load_data(os.path.join(opt.save, 'val_hist.pt'))

    if split == 'train':
        train_hist[epoch] = loss
        torch.save(train_hist, os.path.join(opt.save, 'train_hist.pt'))
        with open(os.path.join(opt.save, 'train_hist'), 'w') as f:
            f.write(utility.hist_to_str(train_hist))
        return train_hist
    elif split == 'val':
        val_hist[epoch] = loss
        torch.save(val_hist, os.path.join(opt.save, 'val_hist.pt'))
        with open(os.path.join(opt.save, 'val_hist'), 'w') as f:
            f.write(utility.hist_to_str(val_hist))
        return val_hist
    else:
        logging.error('Unknown split: ' + split)


class CheckPoint:
    def latest(opt: Namespace) -> Tuple["model", "state"]:
        if opt.resume == None:
            return None, None
        suffix = opt.suffix 
        if opt.suffix == '':
            f = open(os.path.join(opt.resume, 'latest'), 'r')
            suffix = f.read()
        
        checkpoint_path = os.path.join(opt.resume, 'checkpoint' + suffix + '.pt')
        optim_state_path = os.path.join(opt.resume, 'optim_state' + suffix + '.pt')

        print('=> [Resume] Loading checkpoint: ' + checkpoint_path)
        print('=> [Resume] Loading Optim state: ' + optim_state_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        optim_state = torch.load(optim_state_path, map_location='cpu')
        return checkpoint, optim_state

    def save(opt: Namespace, model: CoarseNet, optim_state: dict, epoch: int) -> None:
        checkpoint = {}
        checkpoint['opt'] = opt
        checkpoint['epoch'] = epoch
        checkpoint['model'] = model
        suffix = str(epoch_num)

        if not os.path.exists(opt.save):
            os.makedirs(opt.save)
        
        torch.save(checkpoint, os.path.join(opt.save, 'checkpoint' + suffix + '.pt'))
        torch.save(optim_state, os.path.join(opt.save, 'optim_state' + suffix + '.pt'))

        f = open(os.path.join(opt.save, 'latest'), 'w')
        f.write(suffix)
        f.close()
