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
import matplotlib.pyplot as plt
from math import log, e
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd


parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="Temp")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic1024coarse",type=str,help='File to test on')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")


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
    
dataset = Dataset(os.path.join(input_folder, "JHUturbulence/isotropic1024coarse"), opt)
frame = dataset.__getitem__(0).to("cuda")
frame = dataset.unscale(frame)
p = np2torch(np.load("p.npy"), "cuda")

f = (((frame[:,0,:,:] **2) + (frame[:,1,:,:]**2) + (frame[:,2,:,:]**2))**0.5) #* p[0,:,:,0]
print(f.mean() * 0.5)
#f = frame[:,0,:,:]
bins = torch.histc(f, 100)
plt.plot(np.arange(f.min().item(), f.max().item(), (f.max().item() - f.min().item()) / bins.shape[0]), 
bins.cpu().numpy() / (f.shape[1] * f.shape[2]))
plt.show()
f = torch.cat([f[0].unsqueeze(2), torch.zeros(f[0].shape, device="cuda").unsqueeze(2)], 2)

f_fft = torch.fft(f,2,normalized=True)
"""
f_fft[:,:,0] -= f_fft[:,:,0].min()
f_fft[:,:,1] -= f_fft[:,:,1].min()
f_fft[:,:,0] *= (1/ f_fft[:,:,0].max())
f_fft[:,:,1] *= (1/ f_fft[:,:,1].max())
"""
print(f_fft.shape)
plt.imshow(np.log10(abs(f_fft.cpu().numpy()[:,:,0])))
plt.show()

d = abs(f_fft.cpu().numpy()[:,:,0])[0:int(f_fft.shape[0] / 2), 0:int(f_fft.shape[1]/2)]
spectra = []
ks = []

dis=0.0928
for i in range(d.shape[0]):
    s = np.mean(d[i])
    spectra.append(s)
    if(i == 0):
        ks.append(1.6 * (dis**(2.0/3.0)))
    else:
        ks.append(1.6 * (dis**(2.0/3.0)) * ((i) **(-5.0/3.0)))
plt.plot(np.arange(0, d.shape[0]), spectra)
plt.plot(np.arange(0, d.shape[0]), ks, linestyle="dashed", color="black")

plt.yscale("log")
plt.xscale("log")
plt.legend(["Ground truth", '1.6*e^(2/3)*k^(-5/3)'])
plt.show()