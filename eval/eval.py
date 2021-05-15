#!/usr/bin/env python3

import re
import os
import torch
import torch.nn.functional as F
import struct
import glob
from PIL import Image
import torchvision.transforms.functional as TF
import shutil
from torchvision.utils import save_image

root_dir = ''

def load_flow(filename: str) -> torch.Tensor:
	f = open(filename, 'rb')
	tag = struct.unpack('f', f.read(4))[0]
	
	w = struct.unpack('i', f.read(4))[0]
	h = struct.unpack('i', f.read(4))[0]
	channels = 2
	
	l = [] # in file: [h, w, c]
	for i, val in enumerate(struct.iter_unpack('h', f.read())):
		if not i % 2:
			l.append([])
		l[int(i/2)].append(val[0])
		
	flow = torch.ShortTensor(l).reshape(h, w, channels)
	f.close()
	
	flow = flow.permute(2, 0, 1).float() # output: [c, h, w]
	return flow

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
	
if __name__ == '__main__':
	sub_dir = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
	print(len(sub_dir))
	
	rec_err = 0
	rho_err = 0
	flow_err = 0
	mask_err = 0
	
	count = 0
	
	for d in sub_dir:
		count += 1
		print(count)
		
		bg = glob.glob(os.path.join(root_dir, d, '*bg.png'))[0]
		bg = Image.open(bg)
		bg = TF.to_tensor(bg)
		
		tar = glob.glob(os.path.join(root_dir, d, '*tar.png'))[0]
		tar = Image.open(tar)
		tar = TF.to_tensor(tar)
		
		rec = glob.glob(os.path.join(root_dir, d, '*rec.png'))[0]
		rec = Image.open(rec)
		rec = TF.to_tensor(rec)
		
		mask = glob.glob(os.path.join(root_dir, d, '*mask.png'))[0]
		mask = Image.open(mask)
		mask = TF.to_tensor(mask.convert('L'))
		
		mask_gt = glob.glob(os.path.join(root_dir, d, '*mask_gt.png'))[0]
		mask_gt = Image.open(mask_gt)
		mask_gt = TF.to_tensor(mask_gt.convert('L'))
	
		rho = glob.glob(os.path.join(root_dir, d, '*rho.png'))[0]
		rho = Image.open(rho)
		rho = TF.to_tensor(rho.convert('L'))
		
		rho_gt = glob.glob(os.path.join(root_dir, d, '*rho_gt.png'))[0]
		rho_gt = Image.open(rho_gt)
		rho_gt = TF.to_tensor(rho_gt.convert('L'))
		
		flow = glob.glob(os.path.join(root_dir, d, '*flow.flo'))[0]
		flow = load_flow(flow)
		
		flow_gt = glob.glob(os.path.join(root_dir, d, '*flow_gt.flo'))[0]
		flow_gt = load_flow(flow_gt)
		
		rec_err += 100 * F.mse_loss(rec, tar)
		rho_err += 100 * F.mse_loss(rho, rho_gt)
		flow_err += epe(mask_gt, flow_gt * rho_gt, flow * rho_gt)
		mask_err += iou(mask, mask_gt)

	size = len(sub_dir)
	rec_err /= size
	rho_err /= size
	flow_err /= size
	mask_err /= size
	
	print(f'rec_err: {rec_err}')
	print(f'rho_err: {rho_err}')
	print(f'flow_err: {flow_err}')
	print(f'mask_err: {mask_err}')
