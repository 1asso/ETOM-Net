import torch
import utility
import logging
import os
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torch.utils.data import DataLoader
from models import CoarseNet, RefineNet
from argparse import Namespace
from checkpoint import CheckPoint
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image


class Trainer:
    def __init__(self, model: Union[CoarseNet.CoarseNet, RefineNet.RefineNet], 
    opt: Namespace, optim_state: Optional[dict]) -> None:
        print('\n\n --> Initializing Trainer')
        self.opt = opt
        self.model = model.cuda()
        self.warping_module = self.setup_warping_module() 
        self.multi_scale_data = self.setup_ms_data_module()
        self.optim_state = self.setup_solver(optim_state) 
        self.setup_criterions()
        self.optimizer = torch.optim.Adam(self.model.parameters(), **(self.optim_state), \
                                weight_decay=0.01)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        print('\n\n --> Total number of parameters in ETOM-Net: ' + str(sum(p.numel() for p in self.model.parameters())))

        self.input_image = None
        self.ref_images = None
        self.tar_images = None
        self.rhos = None
        self.masks = None
        self.flows = None
    
    def setup_ms_data_module(self) -> utility.CreateMultiScaleData:
        print('[Multi Scale] Setting up multi scale data module')
        ms_data_module = utility.CreateMultiScaleData(self.opt.ms_num)
        return ms_data_module

    def setup_warping_module(self) -> utility.CreateMultiScaleWarping:
        print('[Multi Scale] Setting up multi scale warping')
        warping_module = utility.CreateMultiScaleWarping(self.opt.ms_num)
        return warping_module

    def setup_criterions(self) -> None:
        print('\n\n --> Setting up criterion')

        print('[Flow Loss] Setting up EPELoss for flow')
        self.flow_criterion = utility.EPELoss

        print('[Rec Loss] Setting up MSELoss for reconstructed image')
        self.rec_criterion = nn.MSELoss
        
        print('[Mask Loss] Setting up CrossENtropyLoss for mask')
        self.mask_criterion = nn.CrossEntropyLoss

        print('[Rho Loss] Setting up MSELoss for rho')
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
        gradient_accumulations = self.opt.ga

        num_batches = len(dataloader)
        print('\n====================')
        print(self.optim_state)
        print(f'Training epoch # {epoch+1}, totaling mini batches {num_batches}')
        print('====================\n')

        self.model.train()
        
        loss_iter = {} # loss every n iterations
        loss_epoch = {} # loss of the entire epoch
        eps = 1e-7

        # Zero gradients
        self.optimizer.zero_grad()

        if self.opt.refine:
            loss_iter['mask'] = 0
            loss_iter['rho'] = 0
            loss_iter['flow'] = 0

            for iter, sample in enumerate(dataloader):
                input = self.setup_inputs(sample)
                
                torch.cuda.empty_cache()
                torch.autograd.set_detect_anomaly(True)

                output = self.model.forward(input)     

                pred_images = self.single_flow_warping(output) # warp input image with flow

                flow_loss = self.flow_criterion()(output[0], self.flows, \
                        self.masks.unsqueeze(1), self.rhos.unsqueeze(1)) 
                mask_loss = self.mask_criterion()(output[1] + eps, self.masks.squeeze(1).long()) 
                rho_loss = self.rho_criterion()(output[2], self.rhos.unsqueeze(1))

                loss = flow_loss + mask_loss + rho_loss
                
                loss_iter['mask'] += mask_loss.item() 
                loss_iter['rho'] += rho_loss.item()
                loss_iter['flow'] += flow_loss.item()

                # Perform a backward pass
                (loss / gradient_accumulations).backward()

                # Update the weights
                if (iter + 1) % gradient_accumulations == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (iter+1) % self.opt.train_display == 0:
                    loss_epoch[iter] = self.display(epoch+1, iter+1, num_batches, loss_iter, split)
                    loss_iter['mask'] = 0
                    loss_iter['rho'] = 0
                    loss_iter['flow'] = 0

                if (iter+1) % self.opt.train_save == 0:
                    self.save_results(epoch+1, iter+1, output, pred_images, split, 0)
        else:
            for i in range(self.opt.ms_num):
                loss_iter[f'Scale {i} mask'] = 0
                loss_iter[f'Scale {i} rho'] = 0
                loss_iter[f'Scale {i} flow'] = 0
                loss_iter[f'Scale {i} rec'] = 0


            for iter, sample in enumerate(dataloader):
                input = self.setup_inputs(sample)
                
                torch.cuda.empty_cache()
                torch.autograd.set_detect_anomaly(True)

                output = self.model.forward(input)     

                pred_images = self.flow_warping(output) # warp input image with flow

                loss = None

                for i in range(self.opt.ms_num):
                    
                    mask_loss = self.opt.mask_w * self.mask_criterion()(output[i][1] + eps, \
                            self.multi_masks[i].squeeze(1).long()) * (1 / 2 ** (self.opt.ms_num - i - 1))
                    rho_loss = self.opt.rho_w * self.rho_criterion()(output[i][2], \
                            self.multi_rhos[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                    flow_loss = self.opt.flow_w * self.flow_criterion()(output[i][0], \
                            self.multi_flows[i], self.multi_masks[i], self.multi_rhos[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                    mask = utility.get_mask(output[i][1]).expand(output[i][1].size(0), \
                            3, output[i][1].size(2), output[i][1].size(3))
                    final_pred = utility.get_final_pred(self.multi_ref_images[i], \
                            pred_images[i], mask, output[i][2])
                    rec_loss = self.opt.img_w * self.rec_criterion()(final_pred, \
                            self.multi_tar_images[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                    
                    if i == 0:
                        loss = mask_loss + rho_loss + flow_loss + rec_loss
                    else:
                        loss += mask_loss + rho_loss + flow_loss + rec_loss
                    
                    loss_iter[f'Scale {i} mask'] += mask_loss.item() 
                    loss_iter[f'Scale {i} rho'] += rho_loss.item()
                    loss_iter[f'Scale {i} flow'] += flow_loss.item()
                    loss_iter[f'Scale {i} rec'] += rec_loss.item()

                # Perform a backward pass
                (loss / gradient_accumulations).backward()
                
                # Update the weights
                if (iter + 1) % gradient_accumulations == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (iter+1) % self.opt.train_display == 0:
                    loss_epoch[iter] = self.display(epoch+1, iter+1, num_batches, loss_iter, split)
                    for i in range(self.opt.ms_num):
                        loss_iter[f'Scale {i} mask'] = 0
                        loss_iter[f'Scale {i} rho'] = 0
                        loss_iter[f'Scale {i} flow'] = 0
                        loss_iter[f'Scale {i} rec'] = 0

                if (iter+1) % self.opt.train_save == 0:
                    self.save_ms_results(epoch+1, iter+1, output, pred_images, split, 0)
        
        average_loss = utility.build_loss_string(utility.dict_of_dict_average(loss_epoch))
        print(f'\n\n --> Epoch: [{epoch+1}] Loss summary: \n{average_loss}')
        self.scheduler.step()
        self.optim_state['lr'] = self.optimizer.param_groups[0]['lr']
        
        return average_loss

    def get_saving_name(self, log_dir: str, split: str, epoch: int, iter: int, id: int) -> str:
        f_path = f'{log_dir}/{split}/Images/'
        f_names = f'epoch:{epoch}_iter:{iter}_id:{id}'
        return os.path.join(f_path, f_names + '.png')

    def save_images(self, pred_images: Tensor, output: List[Tensor], count: int) -> int:
        for i in range(pred_images.size()[0]):
            print(count)
            os.makedirs(f'results/{count}')
            mask = torch.squeeze(utility.get_mask(output[1][i].unsqueeze(0))).expand(3, output[1].size(2), output[1].size(3))
            rho = output[2][i].repeat(3, 1, 1)
            final_img = utility.get_final_pred(self.ref_images[i], pred_images[i], mask, rho)
            save_image(final_img, f'results/{count}/in_rec.png')
            save_image(mask.float(), f'results/{count}/mask.png')
            save_image(rho, f'results/{count}/rho.png')
            utility.save_flow(f'results/{count}/flow.flo', output[0][i])

            save_image(self.ref_images[i], f'results/{count}/bg.png')
            save_image(self.masks[i], f'results/{count}/mask_gt.png')
            save_image(self.rhos[i], f'results/{count}/rho_gt.png')
            save_image(self.input_image[i], f'results/{count}/input.png')
            save_image(self.tar_images[i], f'results/{count}/tar.png')
            utility.save_flow(f'results/{count}/flow_gt.flo', self.flows[i][0:2, :, :])
            save_image(utility.flow_to_color(torch.mul(output[0][i], self.masks[i])), f'results/{count}/fcolor.png')
            save_image(utility.flow_to_color(self.flows[i]), f'results/{count}/fcolor_gt.png')
            count += 1
        return count

    def get_predicts(self, id: int, output: List[Tensor], pred_img: Tensor, m_scale: int) -> List[Tensor]:
        pred = [] 
        if m_scale != None:
            gt_color_flow = utility.flow_to_color(self.multi_flows[m_scale][id])
        else:
            gt_color_flow = utility.flow_to_color(self.flows[id])
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

    def get_first_row(self, id: int) -> List[Union[bool, Tensor]]:
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

        first_row = self.get_first_row(id)
        for val in first_row:
            results.append(val)
        
        for i in range(scales-1, -1, -1):
            sub_pred = self.get_predicts(id, output[i], multi_pred_img[i], i)
            for val in sub_pred:
                results.append(val)
        
        save_name = self.get_saving_name(self.opt.log_dir, split, epoch, iter, id)
        utility.save_compact_results(save_name, results, 6)
        print('\n\n --> Flow magnitude: Max {}, Min {}, Mean {}'.format(
            torch.max(output[scales-1][0][id]), torch.min(output[scales-1][0][id]), 
            torch.mean(torch.abs(output[scales-1][0][id]))))

    def save_results(
        self, 
        epoch: int, 
        iter: int, 
        output: List[Tensor], 
        pred_img: Tensor, 
        split: str, 
        id: int
        ) -> None:
        id = id or 0
        results = []

        first_row = self.get_first_row(id)
        for val in first_row:
            results.append(val)
        
        sub_pred = self.get_predicts(id, output, pred_img, None)
        for val in sub_pred:
            results.append(val)
        
        save_name = self.get_saving_name(self.opt.log_dir, split, epoch, iter, id)
        utility.save_compact_results(save_name, results, 6)

    def flow_warping(self, output: List[List[Tensor]]) -> List[Tensor]:
        flows = []
        for i in range(self.opt.ms_num):
            flows.append(output[i][0])
        
        pred_images= self.warping_module([self.multi_ref_images, flows])
        return pred_images

    def single_flow_warping(self, output: List[Tensor]) -> Tensor:
        pred_images= utility.create_single_warping([self.ref_images, output[0]])
        return pred_images

    def test(self, epoch: int, dataloader: DataLoader, split: str) -> float:
        num_batches = len(dataloader)

        loss_iter = {}
        loss_epoch = {}

        print(f'\n\n===== Testing after {epoch+1} epochs =====')
        
        self.model.eval()
        
        rec_err = 0
        rho_err = 0
        flow_err = 0
        mask_err = 0
        size = 400

        def iou(pred, tar):
            intersection = torch.logical_and(tar, pred)
            union = torch.logical_or(tar, pred)
            iou_score = torch.true_divide(torch.sum(intersection), torch.sum(union))
            return iou_score

        def epe(mask_gt, flow_gt, flow):
            mask_gt = mask_gt.expand_as(flow_gt)
            flow = flow * mask_gt
            flow_gt = flow_gt * mask_gt
            return torch.norm(flow_gt-flow, dim=1).mean() / 100

        if self.opt.refine:
            loss_iter['mask'] = 0
            loss_iter['rho'] = 0
            loss_iter['flow'] = 0

            for iter, sample in enumerate(dataloader):

                with torch.no_grad():
                    input = self.setup_inputs(sample)
                    
                    torch.cuda.empty_cache()
                    torch.autograd.set_detect_anomaly(True)

                    output = self.model.forward(input)     

                    pred_images = self.single_flow_warping(output) # warp input image with flow

                    flow_loss = self.flow_criterion()(output[0], self.flows, \
                            self.masks.unsqueeze(1), self.rhos.unsqueeze(1)) 
                    mask_loss = self.mask_criterion()(output[1], self.masks.squeeze(1).long()) 
                    rho_loss = self.rho_criterion()(output[2], self.rhos.unsqueeze(1))
                    
                    loss_iter['mask'] += mask_loss.item() 
                    loss_iter['rho'] += rho_loss.item()
                    loss_iter['flow'] += flow_loss.item()

                    if (iter+1) % self.opt.val_display == 0:
                        loss_epoch[iter] = self.display(epoch+1, iter+1, num_batches, loss_iter, split)
                        loss_iter['mask'] = 0
                        loss_iter['rho'] = 0
                        loss_iter['flow'] = 0

                    if (iter+1) % self.opt.val_save == 0:
                        self.save_results(epoch+1, iter+1, output, pred_images, split, 0)
        else:
            for i in range(self.opt.ms_num):
                loss_iter[f'Scale {i} mask'] = 0
                loss_iter[f'Scale {i} rho'] = 0
                loss_iter[f'Scale {i} flow'] = 0
                loss_iter[f'Scale {i} rec'] = 0

            count = 1

            for iter, sample in enumerate(dataloader):

                with torch.no_grad():

                    torch.cuda.empty_cache()

                    input = self.setup_inputs(sample)
                    torch.cuda.empty_cache()
                    torch.autograd.set_detect_anomaly(True)

                    output = self.model.forward(input)
                    pred_images = self.flow_warping(output) # warp input image with flow

                    if self.opt.save_images:
                        count = self.save_images(pred_images[-1], output[-1], count)

                    loss = None

                    for i in range(output[-1][0].size(0)):
                        mask = torch.squeeze(utility.get_mask(output[-1][1][i].unsqueeze(0))).expand(3, \
                                output[-1][1][i].size(1), output[-1][1][i].size(2))
                        final_pred = utility.get_final_pred(self.multi_ref_images[-1][i], \
                                pred_images[-1][i], mask, output[-1][2][i])
                        rec_err += 100 * F.mse_loss(final_pred, self.multi_tar_images[-1][i])
                        rho_err += 100 * F.mse_loss(output[-1][2][i], self.multi_rhos[-1][i])
                        flow_err += epe(self.multi_masks[-1][i], self.multi_flows[-1][i][0:2, :, :] * \
                                self.multi_rhos[-1][i], output[-1][0][i] * self.multi_rhos[-1][i])
                        mask_err += iou(mask, self.multi_masks[-1][i])

                    for i in range(self.opt.ms_num):

                        mask_loss = self.opt.mask_w * self.mask_criterion()(output[i][1], \
                                self.multi_masks[i].squeeze(1).long()) * (1 / 2 ** (self.opt.ms_num - i - 1))
                        rho_loss = self.opt.rho_w * self.rho_criterion()(output[i][2], \
                                self.multi_rhos[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                        flow_loss = self.opt.flow_w * self.flow_criterion()(output[i][0], \
                                self.multi_flows[i], self.multi_masks[i], self.multi_rhos[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))
                        mask = utility.get_mask(output[i][1]).expand(output[i][1].size(0), \
                                3, output[i][1].size(2), output[i][1].size(3))
                        final_pred = utility.get_final_pred(self.multi_ref_images[i], \
                                pred_images[i], mask, output[i][2])
                        rec_loss = self.opt.img_w * self.rec_criterion()(final_pred, \
                                self.multi_tar_images[i]) * (1 / 2 ** (self.opt.ms_num - i - 1))

                        loss_iter[f'Scale {i} mask'] += mask_loss.item() 
                        loss_iter[f'Scale {i} rho'] += rho_loss.item()
                        loss_iter[f'Scale {i} flow'] += flow_loss.item()
                        loss_iter[f'Scale {i} rec'] += rec_loss.item()

                    if (iter+1) % self.opt.val_display == 0:
                        loss_epoch[iter] = self.display(epoch+1, iter+1, num_batches, loss_iter, split)
                        for i in range(self.opt.ms_num):
                            loss_iter[f'Scale {i} mask'] = 0
                            loss_iter[f'Scale {i} rho'] = 0
                            loss_iter[f'Scale {i} flow'] = 0
                            loss_iter[f'Scale {i} rec'] = 0

                    if (iter+1) % self.opt.val_save == 0:
                        self.save_ms_results(epoch+1, iter+1, output, pred_images, split, 0)

        rec_err /= size
        rho_err /= size
        flow_err /= size
        mask_err /= size
        eval_str = f'rec_err: {rec_err}\nrho_err: {rho_err}\nflow_err: {flow_err}\nmask_err: {mask_err}\n'
        
        average_loss = utility.build_loss_string(utility.dict_of_dict_average(loss_epoch))
        average_loss = eval_str + average_loss
        print(f'\n\n --> Epoch: [{epoch+1}] Loss summary: \n{average_loss}')
        return average_loss

    def display(self, epoch: int, iter: int, num_batches: int, loss: dict, split: str) -> float:
        interval = (split == 'train') and self.opt.train_display or self.opt.val_display
        average_loss = utility.dict_divide(loss, interval)

        print(f'\n\n --> Epoch ({split}): [{epoch}][{iter}/{num_batches}]')
        print(utility.build_loss_string(average_loss))
        return average_loss

    def setup_inputs(self, sample: dict) -> Tensor:
        self.copy_inputs(sample)
        if not self.opt.refine:
            self.generate_ms_inputs(sample)
            network_input = self.input_image
        else:
            checkpoint = torch.load(self.opt.pred_dir)
            model = checkpoint['model']
            network_input = model.forward(self.input_image)[self.opt.ms_num-1]
            network_input.insert(0, nn.functional.interpolate(
                self.input_image, (512,512), mode='bicubic', align_corners=True))
        
        return network_input

    def copy_inputs(self, sample: dict) -> None:
        del self.ref_images
        del self.tar_images
        del self.masks
        del self.rhos
        del self.flows
        del self.input_image

        self.input_image = torch.cuda.FloatTensor()
        self.ref_images = torch.cuda.FloatTensor()
        self.tar_images = torch.cuda.FloatTensor()
        self.masks = torch.cuda.FloatTensor()
        self.rhos = torch.cuda.FloatTensor()
        self.flows = torch.cuda.FloatTensor()

        n, c, h, w = list(sample['images'].size())
        sh, sw = list(sample['input'].size()[2:])

        self.input_image.resize_(n, 3, sh, sw).copy_(sample['input'])
        self.ref_images.resize_(n, 3, h, w).copy_(sample['images'][:,:3,:,:])
        self.tar_images.resize_(n, 3, h, w).copy_(sample['images'][:,3:,:,:])
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
