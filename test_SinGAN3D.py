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
from skimage.transform.pyramids import pyramid_reduce


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="128_GP_0.5")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic128_3D",type=str,help='File to test on')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')

args = vars(parser.parse_args())

opt = load_options(os.path.join(save_folder, args["load_from"]))
opt["device"] = args["device"]
opt["save_name"] = args["load_from"]
generators, discriminators = load_models(opt,args["device"])

for i in range(len(generators)):
    generators[i] = generators[i].to(args["device"])
    generators[i] = generators[i].eval()
for i in range(len(discriminators)):
    discriminators[i].to(args["device"])
    discriminators[i].eval()

dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)


gen_to_use = 0
lr = opt['resolutions'][gen_to_use]

f = dataset.__getitem__(0).to(opt['device'])

f_hr = f.to(opt['device'])
f_lr = laplace_pyramid_downscale3D(f_hr, opt['n']-gen_to_use-1,
opt['spatial_downscale_ratio'],
opt['device'])

print(f_hr.shape)
print(f_lr.shape)
print(len(generators))

print("SinGAN upscaling:")
singan_output = generate_by_patch(generators, 
"reconstruct", opt, 
opt['device'], opt['patch_size'], 
generated_image=f_lr, start_scale=gen_to_use+1)

print(singan_output.shape)
print("SinGAN error:")
e = ((f_hr - singan_output)**2).mean()
p = PSNR(f_hr, singan_output, f_hr.max() - f_hr.min())
print(e)
print(p)
singan_output = dataset.unscale(singan_output)
singan_output = singan_output.detach().cpu().numpy()[0].swapaxes(0,3).swapaxes(0,1).swapaxes(1,2)
m = np.linalg.norm(singan_output,axis=3)
print(singan_output.shape)
print(m.shape)
from netCDF4 import Dataset
rootgrp = Dataset("singan.nc", "w", format="NETCDF4")
velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
w = rootgrp.createDimension("channels", 3)
us = rootgrp.createVariable("u", singan_output.dtype, ("u","v","w"))
vs = rootgrp.createVariable("v", singan_output.dtype, ("u","v","w"))
ws = rootgrp.createVariable("w", singan_output.dtype, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", singan_output.dtype, ("u","v","w"))
velocities = rootgrp.createVariable("velocities", singan_output.dtype, ("u","v","w", "channels"))
mags[:] = m

print("Trilinear upscaling:")
print(f_lr.shape)
trilin = F.interpolate(f_lr, 
size=generators[-1].resolution, mode=opt["upsample_mode"])
print(trilin.shape)
e = ((f_hr - trilin)**2).mean()
p = PSNR(f_hr, trilin, f_hr.max() - f_hr.min())
print(e)
print(p)
trilin = dataset.unscale(trilin)
trilin = trilin.detach().cpu().numpy()[0].swapaxes(0,3).swapaxes(0,1).swapaxes(1,2)
m = np.linalg.norm(trilin,axis=3)
print(trilin.shape)
print(m.shape)
rootgrp = Dataset("trilinear.nc", "w", format="NETCDF4")
velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
w = rootgrp.createDimension("channels", 3)
us = rootgrp.createVariable("u", trilin.dtype, ("u","v","w"))
vs = rootgrp.createVariable("v", trilin.dtype, ("u","v","w"))
ws = rootgrp.createVariable("w", trilin.dtype, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", trilin.dtype, ("u","v","w"))
velocities = rootgrp.createVariable("velocities", trilin.dtype, ("u","v","w", "channels"))
mags[:] = m


