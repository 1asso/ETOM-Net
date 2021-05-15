#!/usr/bin/env python3

import re
import os
import torch
import torch.nn.functional as F
import struct
import glob
from PIL import Image
import torchvision.transforms.functional as TF
from shutil import copyfile

root_dir = ''
ori_dir = ''

if __name__ == '__main__':
	sub_dir = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
	
	rec_err = 0
	rec_bg = 0
	
	for d in sub_dir:
		print(d)
		
		input = glob.glob(os.path.join(d, '*input.png'))[0]
		name = input[:-10] + '.jpg'
		name = name.split('/')[-1]
		
		ori = glob.glob(os.path.join(ori_dir, name))[0]
		copyfile(ori, input[:-10] + '_tar.png')
		
		
	
	
	
	