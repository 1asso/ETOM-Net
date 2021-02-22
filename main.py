## Import different module, option, dataloader, checkpoint, model, trainer

import torch
import sys
import os
import logging
import numpy as np
from dataloader import create
from checkpoint import CheckPoint
from models.init import setup
from train import Trainer
from option import args
import utility

def init():
    ## Initialization

    # print(args)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_num_threads(1)
    torch.manual_seed(args.manual_seed)

    train_loader, val_loader = create(args)
    check_p, optim_state = CheckPoint.latest(args)
    model = setup(args, check_p)
    trainer = Trainer(model, args, optim_state)
    # torch.backends.cudnn.fastest = True

    if args.val_only:
        results = trainer.test(1, val_loader, 'val')
        return 

    ## Configure start points and history

    train_hist = utility.load_t7(check_p, os.path.join(args.resume and args.resume or '', 'train_hist.t7'))
    val_hist = utility.load_t7(check_p, os.path.join(args.resume and args.resume or '', 'val_hist.t7'))
    start_epoch = check_p.epoch if check_p else args.start_epoch

    def add_history(epoch, loss, split):
        history = {epoch: loss}
        if split == 'train':
            nonlocal train_hist
            train_hist = utility.insert_sub_dict(train_hist, history)
            torch.save(train_hist, os.path.join(args.save, split + '_hist.t7'))
        elif split == 'val':
            nonlocal val_hist
            val_hist = utility.insert_sub_dict(val_hist, history)
            torch.save(val_hist, os.path.join(args.save, split + '_hist.t7'))
        else:
            logging.error('Unknown split: ' + split)

    ## Start training

    for epoch in range(start_epoch, args.n_epochs):
        
        # train for a single epoch
        train_loss = trainer.train(epoch, train_loader, 'train')

        # save checkpoint
        if (epoch+1) % args.save_interval == 0:
            print('\n**** Epoch {} saving checkpoint ****\n'.format(epoch+1))
            CheckPoint.save(args, model, trainer.optim_state, epoch+1)
        


        # save and plot results for training stage
        add_history(epoch+1, train_loss, 'train')
        # utility.plot_results_compact(train_hist, args.log_dir, 'train')

        # validation on synthetic data
        if (epoch+1) % args.val_interval == 0:
            #val_results = trainer.test(epoch, val_loader, 'val')
            val_results = 0
            add_history(epoch+1, val_results, 'val')
            # utility.plot_results_compact(val_hist, args.log_dir, 'val')

if __name__ == '__main__':
    init()
    

