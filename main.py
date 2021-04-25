import torch
import sys
import os
import logging
import numpy as np
from dataloader import create
from checkpoint import CheckPoint, update_history
from models.init import setup
from train import Trainer
from option import args
import utility


if __name__ == "__main__":
    
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_num_threads(1)
    torch.manual_seed(args.manual_seed)

    loaders = create(args)
    check_p, optim_state = CheckPoint.latest(args)
    model = setup(args, check_p)
    trainer = Trainer(model, args, optim_state)

    start_epoch = check_p['epoch'] if check_p else args.start_epoch

    if args.val_only:
        results = trainer.test(0, loaders[1], 'val')
        exit(0)

    for epoch in range(start_epoch, args.n_epochs):
        train_loss = trainer.train(epoch, loaders[0], 'train')
        update_history(args, epoch+1, train_loss, 'train')

        if (epoch+1) % args.save_interval == 0:
            print('\n\n===== Epoch {} saving checkpoint ====='.format(epoch+1))
            CheckPoint.save(args, model, trainer.optim_state, epoch+1)

        if (epoch+1) % args.val_interval == 0:
            val_loss = trainer.test(epoch, loaders[1], 'val')
            update_history(args, epoch+1, val_loss, 'val')
