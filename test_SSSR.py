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

parser.add_argument('--load_from',default="isotropic")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic1024coarse",type=str,help='File to test on')
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
max_mag = dataset.max_mag
num_levels_to_downscale = 7
scaling = 1 / (opt['spatial_downscale_ratio'] ** num_levels_to_downscale)

g = generate(generators, "reconstruct", opt, opt['device'])
g = g.detach().cpu().numpy()[0]
g = to_mag(g,normalize = False)
print(g.shape)
g = toImg(g)
plt.imshow(g.swapaxes(0,2).swapaxes(0,1))
plt.show()
def get_sr_frame(frame):
    o = frame.clone()
    print(o.shape)
    for i in range(num_levels_to_downscale):
        ind = opt['n'] - num_levels_to_downscale + i
        print("Generator %i/%i" % (ind, len(generators)))
        print(generators[ind].resolution)
        o = F.interpolate(o, size=generators[ind].resolution,
        mode=opt["upsample_mode"])
        #noise = torch.randn(o.shape).to(opt['device'])
        noise = generators[ind].optimal_noise
        o = generators[ind](o, opt["noise_amplitudes"][ind]*noise)
    return o

gt_frames = []
bicubic_frames = []
singan_frames = []

bicubic_error = []
singan_error = []
start_frame = 0
print("Scaling ratio; %0.04f" % scaling)
for frame_num in range(start_frame, len(dataset)):
    print("%i/%i" % (frame_num, len(dataset)))
    # Get HR and LR frame
    frame = dataset.__getitem__(frame_num).to(opt['device'])
    f_np = frame.cpu().numpy()[0]

    gt_frames.append(toImg(to_mag(f_np, normalize = False)).swapaxes(0,2).swapaxes(0,1))

    f_np = pyramid_reduce(f_np.swapaxes(0,2).swapaxes(0,1), 
            downscale = scaling,
            multichannel=True).swapaxes(0,1).swapaxes(0,2)
    frame_LR = np2torch(f_np, opt['device']).unsqueeze(0)

    # Create bicubic 
    t = F.interpolate(frame_LR.clone(), 
    size=opt['resolutions'][-1],mode=opt["upsample_mode"])
    t = t[0].detach().cpu().numpy()
    bicubic_frames.append(toImg(to_mag(t, normalize = False)).swapaxes(0,2).swapaxes(0,1))
    bicubic_error.append(((frame.clone().cpu().numpy()-t)**2).mean())

    t = get_sr_frame(frame_LR.clone())
    t = t[0].detach().cpu().numpy()
    singan_frames.append(toImg(to_mag(t, normalize = False)).swapaxes(0,2).swapaxes(0,1))
    singan_error.append(((frame.clone().cpu().numpy()-t)**2).mean())

imageio.mimwrite("GT_frames.gif", gt_frames)
imageio.mimwrite("SinGAN_frames.gif", singan_frames)
imageio.mimwrite("bicubic_frames.gif", bicubic_frames)
plt.plot(range(start_frame, len(dataset)), bicubic_error, color="blue")
plt.plot(range(start_frame, len(dataset)), singan_error, color="red")
plt.legend(['bicubic', 'SinGAN'])
plt.show()