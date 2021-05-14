import os
import torch
from models import CoarseNet, RefineNet
from torch import nn

def setup(opt, checkpoint):
    if checkpoint:
        model = checkpoint['model'].cuda(0)
        return model
    elif opt.retrain:
        assert os.path.exists(opt.retrain), f'Model not found: {opt.retrain}'
        print(f'\n\n --> [Retrain] Loading model from: models/{opt.retrain}')
        model = torch.load(opt.retrain)['model'].cuda(0)
        return model
    elif opt.refine:
        if not opt.val_only:
            print(f'\n\n --> Creating model from: models/RefineNet.py')
            model = RefineNet.RefineNet(opt)
        else:
            print(f'\n\n --> Loading model from: {opt.refine_dir}')
            model = torch.load(opt.refine_dir)['model'].cuda(0)
            return model
    else:
        if not opt.val_only:
            print(f'\n\n --> Creating model from: models/CoarseNet.py')
            model = CoarseNet.CoarseNet(opt)
        else:
            print(f'\n\n --> Loading model from: {opt.pred_dir}')
            model = torch.load(opt.pred_dir)['model'].cuda(0)
            return model

    if torch.cuda.device_count() > 1:
          model = nn.DataParallel(model).cuda(0)
    return model
