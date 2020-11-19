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


a = np.load("0_downsampled.npy")
#a = a[:,::8,::8,::8]

'''
a_0 = laplace_pyramid_downscale3D(np2torch(a[0:1], "cuda:0").unsqueeze(0), 3, 0.5, "cuda:0", periodic=True)
a_1 = laplace_pyramid_downscale3D(np2torch(a[1:2], "cuda:0").unsqueeze(0), 3, 0.5, "cuda:0", periodic=True)
a_2 = laplace_pyramid_downscale3D(np2torch(a[2:3], "cuda:0").unsqueeze(0), 3, 0.5, "cuda:0", periodic=True)
a = torch.cat([a_0, a_1, a_2], axis=1).cpu().numpy()[0]
np.save("0_downsampled_gaussian.npy", a)
'''

from netCDF4 import Dataset
rootgrp = Dataset("isotropic128_downsampled.nc", "w", format="NETCDF4")
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
