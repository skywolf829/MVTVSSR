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


def display_2d(frame):
    scale = 16
    x = np.meshgrid(np.linspace(-1.0, 1.0, int(frame.shape[1]/scale)))
    y = np.meshgrid(np.linspace(-1.0, 1.0, int(frame.shape[2]/scale)))
    plt.quiver(x, y, frame[1,::scale,::scale], frame[0,::scale,::scale], 
    pivot='middle',linestyle='solid')
    plt.show()

def display_2d_mag(frame):
    shape = list(frame.shape)
    shape[0] = 1
    new_chan = np.zeros(shape)
    print(new_chan.shape)
    new_chan[0] = (frame[1]**2 + frame[0]**2)**0.5
    new_chan = new_chan.swapaxes(0,2).swapaxes(0,1)
    plt.imshow(new_chan, cmap="gray")
    plt.show()

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="Temp",help='The type of input - 2D, 3D, or 2D time-varying')
parser.add_argument('--data_folder',default="Synthetic/testing",type=str,help='File to train on')
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

frame_LR = np.load(os.path.join(input_folder, "Synthetic_VFD", "0_256x256.npy"))
print(TAD(np2torch(frame_LR, opt["device"]).unsqueeze(0), opt["device"]))
display_2d(frame_LR)

frame_HR = np.load(os.path.join(input_folder, "Synthetic_VFD HR", "0_512x512.npy"))
print(TAD(np2torch(frame_HR, opt["device"]).unsqueeze(0), opt["device"]))
display_2d_mag(frame_HR)
lr_torch = np2torch(frame_LR, opt["device"]).unsqueeze(0)

t = super_resolution(generators[-1], lr_torch, 2, opt, opt["device"])
t = torch.clamp(t, -1, 1)[0].detach().cpu().numpy()
display_2d_mag(t)
p = 20*log(2) - 10 * log(np.mean((t-frame_HR)**2))
print(p)

lr_2_hr = F.interpolate(lr_torch, size=[512, 512], mode="bicubic")[0].detach().cpu().numpy()
display_2d_mag(lr_2_hr)
p = 20*log(2) - 10 * log(np.mean((lr_2_hr-frame_HR)**2))
print(p)



'''
plt.imshow(frame_LR)
plt.show()
plt.imshow(reconstruct)
plt.show()
plt.imshow(frame_HR)
plt.show()

'''

