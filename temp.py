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
frame = dataset.__getitem__(5).to("cuda")[:,:,::2,::2]



x_particles = 128
y_particles = 128
time_length = 100
ts_per_sec = 2
gt_pathlines = lagrangian_transport(frame, x_particles, y_particles, time_length, ts_per_sec)
#gt_pathline_imgs = viz_pathlines(frame, gt_pathlines, "pathlines_gt", [255, 0, 0])

bicub_errs = []
singan_errs = []
res = []
gen_to_use = 0
for i in range(len(opt['resolutions'])-1):
    gen_to_use = i
    lr = opt['resolutions'][gen_to_use]
    res.append(opt['resolutions'][gen_to_use][0])

    f_lr = pyramid_reduce(frame[0].clone().cpu().numpy().swapaxes(0,2).swapaxes(0,1), 
        downscale = frame.shape[2] / lr[0],
        multichannel=True).swapaxes(0,1).swapaxes(0,2)
    f_lr = np2torch(f_lr, opt['device']).unsqueeze(0).to(opt['device'])

    bicub = F.interpolate(f_lr.clone(), size=opt['resolutions'][-1],mode=opt["upsample_mode"])
    bicub_pathlines = lagrangian_transport(bicub, x_particles, y_particles, time_length, ts_per_sec)
    #bicub_pathline_imgs = viz_pathlines(frame, bicub_pathlines, "pathlines_bicub", [0, 255, 0])

    generated = f_lr.clone()
    for j in range(gen_to_use+1, len(generators)):
        with torch.no_grad():
            generated = F.interpolate(generated, size=opt['resolutions'][j],mode=opt["upsample_mode"]).detach()
            generated = generators[j](generated, 
            #torch.randn(generators[j].get_input_shape()).to(opt["device"]) * opt['noise_amplitudes'][j])
            generators[j].optimal_noise * opt['noise_amplitudes'][j])
    singan_pathlines = lagrangian_transport(generated, x_particles, y_particles, time_length, ts_per_sec)
    #singan_pathline_imgs = viz_pathlines(frame, singan_pathlines, "pathlines_singan", [0, 0, 255])


    singan_pathline_dist = pathline_distance(gt_pathlines, singan_pathlines)
    bicub_pathline_dist = pathline_distance(gt_pathlines, bicub_pathlines)
    singan_errs.append(singan_pathline_dist)
    bicub_errs.append(bicub_pathline_dist)


    #for i in range(len(singan_pathline_imgs)):
    #    singan_pathline_imgs[i] += bicub_pathline_imgs[i] + gt_pathline_imgs[i]
    #imageio.mimwrite("pathlines_all.gif", singan_pathline_imgs)


    #print("Singan pathline list %0.04f" % singan_pathline_dist)
    #print("Bicub pathline list %0.04f" % bicub_pathline_dist)

plt.plot(res, singan_errs, color="red")
plt.plot(res, bicub_errs, color="blue")
plt.title("Pathline error")
plt.legend(["SinGAN", "Bicubic"])
plt.show()
