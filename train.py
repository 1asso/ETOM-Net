import torch
import torch.nn as nn
from easydict import EasyDict as edict
from models.model_utils import ModelUtils
from utils.dict_utils import DictUtils
from utils.str_utils import StrUtils
from utils.io_utils import IOUtils
from utils.flow_utils import FlowUtils
from eval.eval_utils import EvalUtils
from criterion.TOMCriterionFlow import TOMCriterionFlow
from criterion.TOMCriterionUnsup import TOMCriterionUnsup

class Trainer:
    def __init__(self, model, config, optim_state):
        print('Initializing Trainer')
        self.config = config
        self.model = model
        self.warping_module = self.setup_warping(config) # reconstruct input based on refractive flow field
        self.optim_state = self.setup_solver(config, optim_state) # check if resume training
        self.setup_criterion(config)
        if not config.refine:
            # in coarse stage, multi scale ground truth matte is needed
            self.multi_scale_data = self.setup_multi_scale_data(config)
        
        print('Get model parameters and gradient parameters')
        self.params = model.parameters()
        print('Total number of parameters in TOM-Net: ' + str(torch.numel(self.params)))
        # variable to store error for the estimated environment matte
        self.flow_e = 0
        self.mask_e = 0
        self.rho_e = 0
    
    def setup_multi_scale_data(config):
        print('[Multi Scale] Setting up multi scale data')
        # generate multi-scale ground truth during training
        ms_data = ModelUtils.create_multi_scale_data(config)
        return ms_data

    def setup_warping(config):
        print('Setting up warping module')
        if config.refine:
            print('[Single Scale] Setting up single scale warping')
            warping_module = ModelUtils.create_single_warping_module()
            self.c_warp = ModelUtils.create_single_warping_module() # for CoarseNet
        else:
            print('[Multi Scale] Setting up multi scale warping')
            warping_module = ModelUtils.create_multi_scale_data(config.ms_num)
        return warping_module

    def setup_criterion(config):
        print('Setting up criterion')
        print('[Flow Loss] Setting up criterion for flow')
        self.flow_crit = TOMCriterionFlow(config)
        if config.refine:
            # for refinement
            # in refinement stage, an addition flow criterion is initialized
            # to calculate the EPE error for CoarseNet
            self.c_flow_crit = TOMCriterionFlow(config)

            print('[Unsup Loss] Setting up criterion for mask, rho and reconstruction image')
            # criterion for mask, attenuation mask and resconstruction loss
            self.unsup_crit = TOMCriterionUnsup(config)

    def setup_solver(config, in_optim_state):
        optim_state = edict()
        if config.solver == 'ADAM':
            print('[Solver] Using Adam solver')
            optim_state = in_optim_state or {
                'learning_rate': config.lr,
                'beta1': config.beta_1,
                'beta2': config.beta_2
            }
        else:
            logging.warning('Unknown optimization method')

        return optim_state

    def get_refine_input(input, predictor):
        c_ls = edict()
        coarse = predictor.forward(input)

    def test(self, epoch, dataloader, split, *predictor):
        pass

    def train(self, epoch, dataloader, split, *predictor):
        pass
