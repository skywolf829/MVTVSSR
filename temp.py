from SinGAN_models import *
from options import *
from utility_functions import *
import torch.nn.functional as F
import torch
import os
import imageio
import argparse
from typing import Union, Tuple
from matplotlib.pyplot import cm
from math import log
import matplotlib.pyplot as plt




MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

model_name = "iso128_gan"
opt = load_options(os.path.join(save_folder, model_name))
opt['device'] = "cuda:0"
generators, discriminators = load_models(opt,"cuda:0")

out = generate_by_patch(generators, "random", opt, "cuda:0", 128)
print(out.shape)
out = out[0].cpu().numpy()
a = out

from netCDF4 import Dataset
rootgrp = Dataset("iso128_gan.nc", "w", format="NETCDF4")
#velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
c = rootgrp.createDimension("channels", 3)
us = rootgrp.createVariable("u", np.float32, ("u","v","w"))
vs = rootgrp.createVariable("v", np.float32, ("u","v","w"))
ws = rootgrp.createVariable("w", np.float32, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", np.float32, ("u","v","w"))
us[:] = a[0,:,:,:]
vs[:] = a[1,:,:,:]
ws[:] = a[2,:,:,:]

m = np.linalg.norm(a,axis=0)

mags[:] = m
