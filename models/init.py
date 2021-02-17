import os
import torch
import models.CoarseNet
from models import *

def setup(opt, checkpoint):
    if checkpoint:
        model = checkpoint.model
    elif opt.retrain:
        assert os.path.exists(opt.retrain), 'Model not found: {}'.format(opt.retrain)
        print('=> [Retrain] Loading model from: models/{}'.format(opt.retrain))
        model = torch.load(opt.retrain).model.cuda()
    else:
        print('=> Creating model from: models/{}.py'.format(opt.network_type))
        model = getattr(models, opt.network_type)
        model = getattr(model, opt.network_type)(opt)

    return model