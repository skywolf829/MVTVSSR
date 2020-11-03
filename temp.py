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

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="128_GP_0.5")
parser.add_argument('--data_folder',default="JHUturbulence/channel3D",type=str,help='File to test on')
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
a = dataset.__getitem__(0).cuda()
#a = laplace_pyramid_downscale3D(a, 1, 0.5,"cuda")
print(a.shape)
d = TAD3D_CD(a,"cuda")
print(d.shape)
print(d.mean().item())
from netCDF4 import Dataset
rootgrp = Dataset("div.nc", "w", format="NETCDF4")
velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
divergences = rootgrp.createVariable("divergence", np.float32, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", np.float32, ("u","v","w"))

divergences[:] = d[0,0].cpu().numpy()
mags[:] = np.linalg.norm(a[0].cpu().numpy(), axis=0)


'''
a = torch.randn([1, 3, 128, 128, 128], device="cuda")
a = curl3D(a, "cuda")
d = TAD3D_CD(a, "cuda")
print(d.mean().item())
'''