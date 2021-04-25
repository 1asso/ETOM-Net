import os
import torch
from models import CoarseNet, RefineNet
from torch import nn

def setup(opt, checkpoint):
    if checkpoint:
        model = checkpoint['model']
    elif opt.retrain:
        assert os.path.exists(opt.retrain), f'Model not found: {opt.retrain}'
        print(f'\n\n --> [Retrain] Loading model from: models/{opt.retrain}')
        model = torch.load(opt.retrain).model.cuda()
    elif opt.refine:
        print(f'\n\n --> Creating model from: models/RefineNet.py')
        model = RefineNet.RefineNet(opt)
    else:
        print(f'\n\n --> Creating model from: models/CoarseNet.py')
        model = CoarseNet.CoarseNet(opt)

    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model).cuda(0)
    return model
