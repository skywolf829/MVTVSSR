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

parser.add_argument('--load_from',default="isotropic512_magangle")
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
scaling = 4

frame = dataset.__getitem__(0).to(opt['device'])
f_np = frame.cpu().numpy()[0]

imageio.imwrite("GT_HR_mag.png", toImg(to_mag(f_np)).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("GT_HR_uvw.png", toImg(f_np).swapaxes(0,2).swapaxes(0,1))
f_singan = generate(generators, "reconstruct", opt, opt['device'])[0].detach().cpu().numpy()

imageio.imwrite("GT_singan_mag.png", toImg(to_mag(f_singan)).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("GT_singan_uvw.png", toImg(f_singan).swapaxes(0,2).swapaxes(0,1))

'''
max_mag = to_mag(f_np).max()
imageio.imwrite("GT_HR_mag.png", toImg(to_mag(f_np)).swapaxes(0,2).swapaxes(0,1))
f_np = pyramid_reduce(f_np.swapaxes(0,2).swapaxes(0,1), 
        downscale = scaling,
        multichannel=True).swapaxes(0,1).swapaxes(0,2)
imageio.imwrite("GT_LR_mag.png", toImg(to_mag(f_np, max_mag = max_mag)).swapaxes(0,2).swapaxes(0,1))
frame_LR = np2torch(f_np, opt['device']).unsqueeze(0)
#frame_LR = frame[:,:,::4,::4]
print(frame_LR.shape)
#print(TAD(dataset.unscale(frame_LR), opt["device"]).sum())
#plt.imshow(toImg(frame_LR.cpu().numpy()[0], True).swapaxes(0,2).swapaxes(0,1))
#plt.show()

t = F.interpolate(frame_LR, scale_factor=scaling,mode=opt["upsample_mode"])
print(((frame-t)**2).mean())
#print(TAD(dataset.unscale(t), opt["device"]).sum())
t = t[0].detach().cpu().numpy()
imageio.imwrite("Bicubic_mag.png", toImg(to_mag(t, max_mag = max_mag)).swapaxes(0,2).swapaxes(0,1))
#plt.imshow(toImg(t, True).swapaxes(0,2).swapaxes(0,1))
#plt.show()

t = super_resolution(generators[-1], frame_LR, scaling, opt, opt["device"])
print(((frame-t)**2).mean())
#t = generators[-1](t)
#print(((frame-t)**2).mean())
#print(TAD(dataset.unscale(t), opt["device"]).sum())
t = t[0].detach().cpu().numpy()
imageio.imwrite("SinGAN_mag.png", toImg(to_mag(t, max_mag = max_mag)).swapaxes(0,2).swapaxes(0,1))
#plt.imshow(toImg(t, True).swapaxes(0,2).swapaxes(0,1))
#plt.show()


'''