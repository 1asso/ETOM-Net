#!/usr/bin/env python3

from torchvision.utils import save_image
import torch
import glob
import os
import struct
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

root_dir = ''

def load_flow(filename):
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

def create_single_warping(input):
	ref = input[0]
	flo = input[1]
	grid = grid_generator(flo)
	
	output = F.grid_sample(ref, grid, align_corners=True)
	return output

def grid_generator(flow):
	B, C, H, W = flow.size()
	# mesh grid 
	xx = torch.arange(0, W).view(1,-1).repeat(H,1)
	yy = torch.arange(0, H).view(-1,1).repeat(1,W)
	xx = xx.view(1,1,H,W).repeat(B,1,1,1)
	yy = yy.view(1,1,H,W).repeat(B,1,1,1)
	grid = torch.cat((xx,yy),1).float()

	flow = 2.0 * flow.div(H)
	
	flow_clo = flow.clone()
	flow[:,0,:,:] = flow_clo[:,1,:,:]
	flow[:,1,:,:] = flow_clo[:,0,:,:]
	
	# scale grid to [-1,1] 
	grid[:,0,:,:] = 2.0*grid[:,0,:,:].clone() / max(W-1,1)-1.0
	grid[:,1,:,:] = 2.0*grid[:,1,:,:].clone() / max(H-1,1)-1.0
	
	grid = grid + flow
	
	grid = grid.permute(0,2,3,1)
	
	return grid

def get_final_pred(ref_img, pred_img, pred_mask, pred_rho):
	final_pred_img = torch.mul(1 - pred_mask, ref_img) + torch.mul(pred_mask, torch.mul(pred_img, pred_rho))
	return final_pred_img

bg = glob.glob(os.path.join(root_dir, '*bg.png'))[0]
bg = Image.open(bg)
bg = TF.to_tensor(bg)

mask = glob.glob(os.path.join(root_dir, '*mask.png'))[0]
mask = Image.open(mask)
mask = TF.to_tensor(mask.convert('L'))
mask.apply_(lambda x: 1 if x else 0)

rho = glob.glob(os.path.join(root_dir, '*rho.png'))[0]
rho = Image.open(rho)
rho = TF.to_tensor(rho.convert('L'))

flow = glob.glob(os.path.join(root_dir, '*flow.png'))[0]
flow = load_flow(flow)

pred = create_single_warping([bg.unsqueeze(0), flow.unsqueeze(0)])

final = get_final_pred(bg, pred, mask, rho)

save_image(final, root_dir + 'rec.png')

print('done')
