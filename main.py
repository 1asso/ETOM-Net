## Import different module, options, dataloader, checkpoints, model, trainer

import torch
import sys
import numpy as np
from dataloader import DataLoader
from checkpoints import CheckPoints
from models.init import setup
from train import Trainer
from config import config


## Initialization

args = sys.argv[1:]
while args:
    key = args.pop(0)[1:]
    config[key] = args.pop(0)

torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(1)
torch.manual_seed(config.manual_seed)

train_loader, val_loader = DataLoader.create(config)
check_p, optim_state = CheckPoints.latest(config)
model = setup(config, check_p)
trainer = Trainer(model, config, optim_state)
# torch.backends.cudnn.fastest = True
