import os
import torch
import torch.nn as nn
import math

class CheckPoint:
    def latest(opt):
        if opt.resume == None:
            return None, None
        suffix = opt.suffix  # if specify checkpoint epoch number
        if opt.suffix == '':
            f = open(os.path.join(opt.resume, 'latest'), 'r')
            suffix = f.read()
        
        checkpoint_path = os.path.join(opt.resume, 'checkpoint', suffix, '.pt')
        optim_state_path = os.path.join(opt.resume, 'optim_state', suffix, '.pt')

        print('=> [Resume] Loading checkpoint: ' + checkpoint_path)
        print('=> [Resume] Loading Optim state: ' + optim_state_path)
        checkpoint = torch.load(checkpoint_path)
        optim_state = torch.load(optim_state_path)
        return checkpoint, optim_state

    def save(opt, model, optim_state, epoch):
        #  create a clean copy on the CPU without modifying the original network
        checkpoint = {}
        checkpoint['opt'] = opt
        checkpoint['epoch'] = epoch
        checkpoint['model'] = model
        if opt.save_new > 0:
            epoch_num = math.floor((epoch - 1) / opt.save_new) * opt.save_new + 1
            suffix = str(epoch_num)
        else:
            suffix = ''

        if not os.path.exists(opt.save):
            os.makedirs(opt.save)
        
        torch.save(checkpoint, os.path.join(opt.save, 'checkpoint' + suffix + '.pt'))
        torch.save(optim_state, os.path.join(opt.save, 'optim_state' + suffix + '.pt'))

        f = open(os.path.join(opt.save, 'latest'), 'w')
        f.write(suffix)
        f.close()

