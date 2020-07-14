from easydict import EasyDict as edict
import os
import torch
import torch.nn as nn
import math

class CheckPoints:
    def latest(config):
        if config.resume == 'none':
            return None
        suffix = config.suffix  # if specify checkpoint epoch number
        if config.suffix == '':
            f = open(os.path.join(config.resume, 'latest'), 'r')
            suffix = f.read()
        
        checkpoint_path = os.path.join(config.resume, 'checkpoint', suffix, '.t7')
        optim_state_path = os.path.join(config.resume, 'optim_state', suffix, '.t7')

        print('=> [Resume] Loading checkpoint: ' + checkpoint_path)
        print('=> [Resume] Loading Optim state: ' + optim_state_path)
        checkpoint = torch.load(checkpoint_path)
        optim_state = torch.load(optim_state_path)
        return checkpoint, optim_state

    def save(config, model, optim_state, epoch):
        #  create a clean copy on the CPU without modifying the original network

        checkpoint = edict()
        checkpoint.config = config
        checkpoint.epoch = epoch
        checkpoint.model = model
        if config.save_new > 0:
            epoch_num = math.floor((epoch - 1) / config.save_new) * config.save_new + 1
            suffix = str(epoch_num)
        else:
            suffix = ''
        
        torch.save(os.path.join(config.save, 'checkpoint', suffix, '.t7'), checkpoint)
        torch.save(os.path.join('optim_state', suffix, '.t7'), optim_state)

        f = open(os.path.join(config.save, 'latest'), 'w')
        f.write(suffix)
        f.close()

