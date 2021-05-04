import os
import argparse
import numpy as np
import time
import datetime
from typing import Tuple
import torch

def get_save_dir_name(args: argparse.Namespace) -> Tuple[str, str]:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    d_name = date + '_' + (args.refine and 'RefineNet' or 'CoarseNet')

    params = ['flow_w', 'mask_w', 'rho_w', 'img_w', 'lr']
    for p in params:
        d_name = d_name + '_' + p + '-' + str(vars(args)[p])

    d_name = d_name + ('_retrain' if args.retrain != None else '')
    d_name = d_name + ('_resume' if args.resume != None else '')
    d_name = d_name + ('_valOnly' if args.val_only else '')

    log_dir = os.path.join('data/training', d_name, 'logdir')
    save = os.path.join('data/training', d_name, 'checkpointdir')

    return log_dir, save

parser = argparse.ArgumentParser(description='ETOM-Net')

# dataset options
parser.add_argument('--dataset', type=str, default='TOMDataset',
                    help='dataset name')
parser.add_argument('--data_dir', type=str, default='../TOM-Net_Synth_Train_178k',
                    help='training dataset path')
parser.add_argument('--train_list', type=str, default='train_60k.txt',
                    help='train list')
parser.add_argument('--val_list', type=str, default='val_400.txt',
                    help='val list')
parser.add_argument('--data_aug', type=bool, default=True,
                    help='data augmentation')
parser.add_argument('--max_train_num', type=int, default=-1,
                    help='>0 for max number')
parser.add_argument('--max_val_num', type=int, default=-1,
                    help='>0 for max number')
parser.add_argument('--processes', type=int, default=16,
                    help='number of data loading processes')

# training options
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of total epochs to run')
parser.add_argument('--ga', type=int, default=1,
                    help='gradient accumulations')
parser.add_argument('--batch_size', type=int, default=16,
                    help='mini-batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--solver', type=str, default='ADAM',
                    help='solver used(Adam only)')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='first param of Adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.999,
                    help='second param of Adam optimizer')

# network options
parser.add_argument('--ms_num', type=int, default=4,
                    help='multiscale level')
parser.add_argument('--refine', action='store_true',
                    help='train refine net')
parser.add_argument('--pred_dir', type=str, default='coarse.pt',
                    help='predictor path')

# checkpoint options
parser.add_argument('--resume', type=str, default=None,
                    help='reload checkpoint and state')
parser.add_argument('--retrain', type=str, default=None,
                    help='reload checkpoint only')
parser.add_argument('--save_interval', type=int, default=1,
                    help='epochs to save checkpoint(overwrite)')

# loss options
parser.add_argument('--flow_w', type=float, default=0.01,
                    help='flow weight')
parser.add_argument('--img_w', type=int, default=1,
                    help='image reconstruction weight')
parser.add_argument('--mask_w', type=float, default=0.3,
                    help='mask weight')
parser.add_argument('--rho_w', type=int, default=3,
                    help='attenuation mask weight')

# display options
parser.add_argument('--train_display', type=int, default=10,
                    help='iteration to display train loss')
parser.add_argument('--train_save', type=int, default=100,
                    help='iteration to save train results')
parser.add_argument('--val_interval', type=int, default=1,
                    help='epoch to do validation')
parser.add_argument('--val_display', type=int, default=1,
                    help='iteration to display val loss')
parser.add_argument('--val_save', type=int, default=1,
                    help='iteration to save val results')
parser.add_argument('--val_only', action='store_true',
                    help='run on validation set only')

args = parser.parse_args()

args.batch_size *= torch.cuda.device_count()
print("\n\n --> Let's use", torch.cuda.device_count(), "GPUs!")

args.log_dir, args.save = get_save_dir_name(args)

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.isdir(args.save):
    os.makedirs(args.save)
