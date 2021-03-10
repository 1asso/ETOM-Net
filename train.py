import torch
import utility
import logging
import os
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.data import DataLoader
from models import CoarseNet
from argparse import Namespace

class Trainer:
    def __init__(self, model: CoarseNet, opt: Namespace, optim_state: Optional[dict]) -> None:
        print('Initializing Trainer')
        self.opt = opt
        self.model = model.cuda()
        self.warping_module = self.setup_warping_module() # reconstruct input based on refractive flow field
        self.optim_state = self.setup_solver(optim_state) # check if resume training
        self.setup_criterions()
        self.multi_scale_data = self.setup_ms_data_module()
        
        print('Total number of parameters in ETOM-Net: ' + str(sum(p.numel() for p in self.model.parameters())))

        self.ref_images = None
        self.tar_images = None
        self.rhos = None
        self.masks = None
        self.flows = None
    
    def setup_ms_data_module(self) -> utility.CreateMultiScaleData:
        print('[Multi Scale] Setting up multi scale data module')
        # generate multi-scale ground truth during training
        ms_data_module = utility.CreateMultiScaleData(self.opt.ms_num)
        return ms_data_module

    def setup_warping_module(self) -> utility.CreateMultiScaleWarping:
        print('[Multi Scale] Setting up multi scale warping')
        warping_module = utility.CreateMultiScaleWarping(self.opt.ms_num)
        return warping_module

    def setup_criterions(self) -> None:
        print('Setting up criterion')
        print('[Flow Loss] Setting up criterion for flow')
        self.flow_criterion = utility.EPELoss
        print('[Unsup Loss] Setting up criterion for mask, rho and reconstruction image')
        # criterion for mask, attenuation mask and resconstruction loss
        self.rec_criterion = nn.MSELoss
        self.mask_criterion = nn.CrossEntropyLoss
        self.rho_criterion = nn.MSELoss

    def setup_solver(self, in_optim_state: dict) -> dict:
        optim_state = None
        if self.opt.solver == 'ADAM':
            print('[Solver] Using Adam solver')
            optim_state = in_optim_state or {
                'lr': self.opt.lr,
                'betas': (self.opt.beta_1, self.opt.beta_2)
            }
        else:
            logging.warning('Unknown optimization method')

        return optim_state

    def train(self, epoch: int, dataloader: DataLoader, split: str) -> float:
        torch.cuda.empty_cache()
        
        gradient_accumulations = 1

        split = split or 'train'
        self.optim_state['lr'] = self.update_lr(epoch)
        print('Epoch {}, Learning rate {}'.format(epoch+1, self.optim_state['lr']))
        num_batches = len(dataloader)
        print('====================')
        print(self.optim_state)
        print('====================')
        print('Training epoch # {}, totaling mini batches {}'.format(epoch+1, num_batches))

        self.model.train()
        
        loss_iter = {} # loss every 20 iterations
        loss_epoch = {} # loss of entire epochs

        for i in range(self.opt.ms_num):
            loss_iter[f'Scale {i}: mask'] = 0
            loss_iter[f'Scale {i}: rho'] = 0
            loss_iter[f'Scale {i}: flow'] = 0
            loss_iter[f'Scale {i}: rec'] = 0

        optimizer = torch.optim.Adam(self.model.parameters(), **(self.optim_state))

        # Zero gradients
        optimizer.zero_grad()

        for iter, sample in enumerate(dataloader):
            input = self.setup_inputs(sample)
            
            torch.cuda.empty_cache()
            torch.autograd.set_detect_anomaly(True)

            output = self.model.forward(input)            

            pred_images = self.flow_warping(output) # warp input image with flow

            loss = None

            for i in range(self.opt.ms_num):

                mask_loss = self.opt.mask_w * self.mask_criterion()(output[i][1], self.multi_masks[i].squeeze(1).long()) * (1 / 2 ** (self.opt.ms_num - i - 1))
                rho_loss = self.opt.rho_w * self.rho_criterion()(output[i][2], self.multi_rhos[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                flow_loss = self.opt.flow_w * self.flow_criterion()(output[i][0], self.multi_flows[i], self.multi_masks[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                rec_loss = self.opt.img_w * self.rec_criterion()(pred_images[i], self.multi_tar_images[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))

                if i == 0:
                    loss = mask_loss + rho_loss + flow_loss + rec_loss
                else:
                    loss += mask_loss + rho_loss + flow_loss + rec_loss
                
                loss_iter[f'Scale {i}: mask'] += mask_loss.item() 
                loss_iter[f'Scale {i}: rho'] += rho_loss.item()
                loss_iter[f'Scale {i}: flow'] += flow_loss.item()
                loss_iter[f'Scale {i}: rec'] += rec_loss.item()

            # Perform a backward pass
            (loss / gradient_accumulations).backward()

            # Update the weights
            if (iter + 1) % gradient_accumulations == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (iter+1) % self.opt.train_display == 0:
                loss_epoch[iter] = self.display(epoch+1, iter+1, num_batches, loss_iter, split)
                for i in range(self.opt.ms_num):
                    loss_iter[f'Scale {i}: mask'] = 0
                    loss_iter[f'Scale {i}: rho'] = 0
                    loss_iter[f'Scale {i}: flow'] = 0
                    loss_iter[f'Scale {i}: rec'] = 0

            if (iter+1) % self.opt.train_save == 0:
                self.save_ms_results(epoch+1, iter+1, output, pred_images, split, 0)
        
        average_loss = utility.dict_of_dict_average(loss_epoch)
        #for name, params in self.model.named_parameters():
	    #    print('-->name:', name, '   -->weight', torch.mean(params.data))
        print('\n\n | Epoch: [{}] Losses summary: {}'.format(epoch+1, utility.build_loss_string(average_loss)))
        return average_loss

    def get_saving_name(self, log_dir: str, split: str, epoch: int, iter: int, id: int) -> str:
        f_path = f'{log_dir}/{split}/Images/'
        f_names = f'epoch:{epoch}_iter:{iter}_id:{id}'
        return os.path.join(f_path, f_names + '.png')

    def get_predicts(self, split: str, id: int, output: List[Tensor], pred_img: Tensor, m_scale: int) -> List[Tensor]:
        pred = [] 
        gt_color_flow = utility.flow_to_color(self.multi_flows[m_scale][id])
        pred.append(gt_color_flow)

        color_flow = utility.flow_to_color(output[0][id])
        pred.append(color_flow)
        mask = torch.squeeze(utility.get_mask(output[1][id].unsqueeze(0))).expand(3, output[1].size(2), output[1].size(3))
        pred.append(mask)
        rho = output[2][id].repeat(3, 1, 1)
        pred.append(rho)

        if m_scale != None:
            final_img = utility.get_final_pred(self.multi_ref_images[m_scale][id], pred_img[id], mask, rho)
            first_img = self.multi_tar_images[m_scale][id]
        else:
            final_img = utility.get_final_pred(self.ref_images[id], pred_img[id], mask, rho)
            first_img = self.tar_images[id]

        pred.insert(0, first_img)
        pred.insert(1, final_img)
        return pred

    def get_first_row(self, split: str, id: int) -> List[Union[bool, Tensor]]:
        first = []
        first.append(self.ref_images[id])
        first.append(self.tar_images[id])
        first.append(False)
        first.append(False)
        first.append(self.masks[id])
        first.append(self.rhos[id])
        return first

    def save_ms_results(
        self, 
        epoch: int, 
        iter: int, 
        output: List[List[Tensor]], 
        multi_pred_img: List[List[Tensor]], 
        split: str, 
        id: int
        ) -> None:
        id = id or 0
        scales = self.opt.ms_num
        results = []

        first_row = self.get_first_row(split, id)
        for val in first_row:
            results.append(val)
        
        for i in range(scales-1, -1, -1):
            sub_pred = self.get_predicts(split, id, output[i], multi_pred_img[i], i)
            for val in sub_pred:
                results.append(val)
        
        save_name = self.get_saving_name(self.opt.log_dir, split, epoch, iter, id)
        utility.save_compact_results(save_name, results, 6)
        print('\n\n | Flow magnitude: Max {}, Min {}, Mean {}'.format(
            torch.max(output[scales-1][0][id]), torch.min(output[scales-1][0][id]), 
            torch.mean(torch.abs(output[scales-1][0][id]))))

    # for image reconstruction loss and image warping
    def flow_warping(self, output: List[List[Tensor]]) -> List[Tensor]:
        flows = []
        for i in range(self.opt.ms_num):
            flows.append(output[i][0])
        
        pred_images= self.warping_module([self.multi_ref_images, flows])
        return pred_images

    def test(self, epoch: int, dataloader: DataLoader, split: str) -> float:
        num_batches = dataloader.get_num_of_batches()

        loss = []
        losses = []  # loss in the entire epoch

        print('*** Testing after {} epochs ***'.format(epoch))
        self.model.evaluate()

        for i, sample in enumerate(dataloader.run(split)):
            input = self.setup_inputs(sample)

            output = self.model.forward(input)

            flows, pred_images= self.flow_warping(output)

            unsup_loss = self.unsup_crit_forward_backward(output, pred_images, True)
            utility.dicts_add(loss, unsup_loss)

            sup_loss = self.sup_crit_forward_backward(flows, True)
            utility.dicts_add(loss, sup_loss)

            val_disp = (split == 'val') and (iter % self.opt.val_display) == 0
            if val_disp:
                losses[iter] = self.display(epoch, iter, num_batches, loss, split)
                utility.dict_reset(loss)
            
            val_save = (split == 'val') and (iter % self.opt.val_save) == 0

            if val_save:
                self.save_ms_results(epoch, iter, output, pred_images, split)
            
        average_loss = utility.dict_of_dict_average(losses)
        print('\n\n | Epoch: [{}] Losses summary: {}'.format(epoch, utility.build_loss_string(average_loss)))
        return average_loss

    def display(self, epoch: int, iter: int, num_batches: int, loss: dict, split: str) -> float:
        interval = (split == 'train') and self.opt.train_display or self.opt.val_display
        average_loss = utility.dict_divide(loss, interval)

        print('\n\n | Epoch ({}): [{}][{}/{}]'.\
            format(split, epoch, iter, num_batches))
        print(utility.build_loss_string(average_loss))
        return average_loss

    def setup_inputs(self, sample: dict) -> Tensor:
        self.copy_inputs(sample)
        self.generate_ms_inputs(sample)

        network_input = self.tar_images
        
        return network_input

    def copy_inputs(self, sample: dict) -> None:
        del self.ref_images
        del self.tar_images
        del self.masks
        del self.rhos
        del self.flows

        self.ref_images = torch.cuda.FloatTensor()
        self.tar_images = torch.cuda.FloatTensor()
        self.masks = torch.cuda.FloatTensor()
        self.rhos = torch.cuda.FloatTensor()
        self.flows = torch.cuda.FloatTensor()

        n, c, h, w = list(sample['input'].size())
        
        self.ref_images.resize_(n, 3, h, w).copy_(sample['input'][:,:3,:,:])
        self.tar_images.resize_(n, 3, h, w).copy_(sample['input'][:,3:,:,:])
        self.masks.resize_(n, h, w).copy_(sample['masks'])
        self.rhos.resize_(n, h, w).copy_(sample['rhos'])
        self.flows.resize_(n, 3, h, w).copy_(sample['flows'])

    def generate_ms_inputs(self, sample: dict) -> None:
        multiscale_in = [self.ref_images, self.tar_images, self.rhos, self.masks, self.flows]

        multiscale_out = self.multi_scale_data(multiscale_in)
        self.multi_ref_images = multiscale_out[0]
        self.multi_tar_images = multiscale_out[1]
        self.multi_rhos = multiscale_out[2]
        self.multi_masks = multiscale_out[3]
        self.multi_flows = multiscale_out[4]

        for i in range(self.opt.ms_num):
            # rescale the loss weight for flow in different scale
            ratio = 2 ** (self.opt.ms_num - i - 1)
            self.multi_flows[i][:, 2, :] *= ratio

            self.multi_masks[i] = self.multi_masks[i].unsqueeze(1)
            self.multi_rhos[i] = self.multi_rhos[i].unsqueeze(1)

    def update_lr(self, epoch: int) -> float:
        ratio = (epoch >= self.opt.lr_decay_start and \
            epoch % self.opt.lr_decay_step == 0) and 0.5 or 1.0
        return self.optim_state['lr'] * ratio