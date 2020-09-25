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

parser.add_argument('--load_from',default="Temp")
parser.add_argument('--data_folder',default="first_sim",type=str,help='File to test on')
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

dataset = Dataset(os.path.join(input_folder, 'first_sim'), opt)

frame_LR = dataset.__getitem__(0).to(opt['device'])
print(frame_LR.shape)
print(TAD(dataset.unscale(frame_LR), opt["device"]).sum())
plt.imshow(toImg(frame_LR.cpu().numpy()[0]).swapaxes(0,2).swapaxes(0,1))
plt.show()

scaling = 2
t = F.interpolate(frame_LR, scale_factor=scaling,mode=opt["upsample_mode"])
print(TAD(dataset.unscale(t), opt["device"]).sum())
t = t[0].detach().cpu().numpy()
plt.imshow(toImg(t).swapaxes(0,2).swapaxes(0,1))
plt.show()

t = super_resolution(generators[-1], frame_LR, scaling, opt, opt["device"])
print(TAD(dataset.unscale(t), opt["device"]).sum())
t = torch.clamp(t, -1, 1)[0].detach().cpu().numpy()
plt.imshow(toImg(t).swapaxes(0,2).swapaxes(0,1))
plt.show()


