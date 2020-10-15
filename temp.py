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

parser.add_argument('--load_from',default="Temp")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic512coarse",type=str,help='File to test on')
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



f = dataset.__getitem__(0).to(opt['device'])
print(f.min())
print(f.max())
#singan_generated_patch = generate_by_patch(generators, 'reconstruct', opt, opt['device'], opt['patch_size'])

singan_generated = generate(generators, 'reconstruct', opt, opt['device'])
rec_numpy = singan_generated.detach().cpu().numpy()[0]
rec_cm = toImg(rec_numpy)
print(rec_cm.shape)
imageio.imwrite("full.png", rec_cm.swapaxes(0,2).swapaxes(0,1))
#imageio.imwrite("full_gt.png", f.detach().cpu().numpy()[0].swapaxes(0,2).swapaxes(0,1))
print(singan_generated.min())
print(singan_generated.max())