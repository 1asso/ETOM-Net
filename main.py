## Import different module, options, dataloader, checkpoints, model, trainer

import torch
import sys
import os
import logging
import numpy as np
from easydict import EasyDict as edict
from dataloader import DataLoader
from checkpoints import CheckPoints
from models.init import setup
from train import Trainer
from config import Config
from utils.dict_utils import DictUtils
from utils.io_utils import IOUtils

def init():
    ## Initialization

    args = sys.argv[1:]
    config = Config.parse(args)

    print(config)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_num_threads(1)
    torch.manual_seed(config.manual_seed)

    train_loader, val_loader = DataLoader.create(config)
    check_p, optim_state = CheckPoints.latest(config)
    model = setup(config, check_p)
    trainer = Trainer(model, config, optim_state)
    # torch.backends.cudnn.fastest = True

    if config.val_only:
        results = trainer.test(1, val_loader, 'val')
        return 

    ## Configure start points and history

    train_hist = IOUtils.load_t7(check_p, os.path.join(config.resume, 'train_hist.t7'))
    val_hist = IOUtils.load_t7(check_p, os.path.join(config.resume, 'val_hist.t7'))
    start_epoch = check_p.epoch + 1 if check_p else config.start_epoch

    def add_history(epoch, history, split):
        if split == 'train':
            nonlocal train_hist
            train_hist = DictUtils.insert_sub_dict(train_hist, history)
            torch.save(os.path.join(config.save, split + '_hist.t7'), train_hist)
        if split == 'val':
            nonlocal val_hist
            val_hist = DictUtils.insert_sub_dict(val_hist, history)
            torch.save(os.path.join(config.save, split + '_hist.t7'), val_hist)
        else:
            logger.error('Unknown split:' + split)

    ## Start training

    for epoch in range(start_epoch, config.n_epochs):
        
        # train for a single epoch
        train_loss = trainer.train(epoch, train_loader, 'train')

        # save checkpoints
        if epoch % config.save_interval == 0:
            print('**** Epoch {} saving checkpoint ****'.format(epoch))
            CheckPoints.save(config, model, trainer.optim_state, epoch)

        # save and plot results for training stage
        add_history(epoch, train_loss, 'train')
        # IOUtils.plot_results_compact(train_hist, config.log_dir, 'train')

        # validation on synthetic data
        if epoch % config.val_interval == 0:
            val_results = trainer.test(epoch, val_loader, 'val')
            add_history(epoch, val_results, 'val')
            # IOUtils.plot_results_compact(val_hist, config.log_dir, 'val')

if __name__ == '__main__':
    init()
    

