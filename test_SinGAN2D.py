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
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.transform.pyramids import pyramid_reduce
from energy_spectra_analysis import *
from scipy import stats

def save_VF(vf, name, error_field=None, streamline_error_field=None, 
particle_seeds=None, angle_error_field=None, magnitude_error_field=None,
vorticity_field=None):
    from netCDF4 import Dataset
    rootgrp = Dataset(name+".nc", "w", format="NETCDF4")
    
    u = rootgrp.createDimension("u")
    v = rootgrp.createDimension("v")
    c = rootgrp.createDimension("channels", 3)
    us = rootgrp.createVariable("u", np.float32, ("u","v"))
    vs = rootgrp.createVariable("v", np.float32, ("u","v"))
    ws = rootgrp.createVariable("w", np.float32, ("u","v"))
    mags = rootgrp.createVariable("magnitude", np.float32, ("u","v"))
    us[:] = vf[0,:,:].cpu().numpy()
    vs[:] = vf[1,:,:].cpu().numpy()
    ws[:] = vf[2,:,:].cpu().numpy()

    if(error_field is not None):
        errs = rootgrp.createVariable("err", np.float32, ("u","v"))
        errs[:] = error_field
    if(streamline_error_field is not None):
        stream_errs = rootgrp.createVariable("streamline_err", np.float32, ("u","v"))
        stream_errs[:] = streamline_error_field
    if(particle_seeds is not None):
        seeds = rootgrp.createVariable("seeds", np.float32, ("u","v"))
        seeds[:] = particle_seeds
    if(angle_error_field is not None):
        anglerr = rootgrp.createVariable("angle_err", np.float32, ("u","v"))
        anglerr[:] = angle_error_field
    if(magnitude_error_field is not None):
        magerr = rootgrp.createVariable("mag_err", np.float32, ("u","v"))
        magerr[:] = magnitude_error_field
    if(vorticity_field is not None):
        vortmag = rootgrp.createVariable("vorticity_magnitude", np.float32, ("u","v"))
        vortmag[:] = vorticity_field

    m = np.linalg.norm(vf.cpu().numpy(),axis=0)

    mags[:] = m

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="0.5_2_zero")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic128_downsampled",
type=str,help='File to test on')
parser.add_argument('--device',default="cuda:0",type=str,
help='Frames to use from training file')

args = vars(parser.parse_args())

opt = load_options(os.path.join(save_folder, args["load_from"]))
opt["device"] = args["device"]


dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)

test_data_folder = "TestingData/iso128/"
start_ts = 0
end_ts = 99
ts_skip = 1
ts_to_save = 475

models_to_try = [
    #"0.5_2_zero",
    #"0.5_2_noise",
    #"0.5_4_zero",
    #"0.5_4_noise",
    #"0.5_8_zero",
    #"0.5_8_noise",
    #"0.5_16_zero",
    #"0.5_16_noise",
    #"0.5_32_zero",
    #"0.5_32_noise",
    #"0.5_64_zero",
    #"0.5_64_noise",
    #"0.5_128_zero",
    #"0.5_128_noise",
    "0.758_2_zero",
    #"0.758_2_noise",
    "0.758_4_zero",
    #"0.758_4_noise",
    "0.758_8_zero",
    #"0.758_8_noise",
    "0.758_16_zero",
    #"0.758_16_noise",
    "0.758_32_zero",
    #"0.758_32_noise",
    "0.758_64_zero",
    #"0.758_64_noise",
    "0.758_128_zero",
    #"0.758_128_noise"
]

streamline_errors = []
PSNRs = []
MSEs = []
energy_spectra = []
labels = []

for model_name in models_to_try:
    print(model_name)
    labels.append(model_name)
    opt = load_options(os.path.join(save_folder, model_name))
    opt['device'] = "cuda:0"
    generators, discriminators = load_models(opt,args["device"])

    for i in range(len(generators)):
        generators[i] = generators[i].to(args["device"])
        generators[i] = generators[i].eval()
    for i in range(len(discriminators)):
        discriminators[i].to(args["device"])
        discriminators[i].eval()
    
    ps = []
    mses = []
    streams = []
    num_ts = 0
    for ts in range(start_ts, end_ts, ts_skip):
        print(ts)
        data = np.load(test_data_folder + str(ts)+".npy")
        data = data[:,:,:,int(data.shape[3]/2)]
        data = np2torch(data, "cuda:0").unsqueeze(0)
        data = dataset.scale(data)

        data_lr = laplace_pyramid_downscale2D(data, opt['n']-1,
            opt['spatial_downscale_ratio'],
            "cuda:0", opt['periodic'])
        
        upscaled_data = generate_by_patch(generators, 
        "reconstruct", opt, 
        opt['device'], opt['patch_size'], 
        generated_image=data_lr, start_scale=1)

        p = PSNR(data, upscaled_data, data.max() - data.min())
        m = torch.abs(upscaled_data - data)

        cs = torch.nn.CosineSimilarity(dim=1).to(opt['device'])
        mags = torch.abs(torch.norm(upscaled_data, dim=1) - torch.norm(data, dim=1))
        angles = torch.abs(cs(upscaled_data, data) - 1) / 2
        upscaled_data = dataset.unscale(upscaled_data)
            
        ps.append(p)

        num_ts += 1
       
        if(ts == ts_to_save):
            vort = curl2D(data, opt['device'])
            vort_mag = torch.norm(vort, dim=1)[0]
            save_VF(upscaled_data[0], model_name, 
            error_field=(mags+angles)[0].cpu().numpy(), 
            angle_error_field=angles[0].cpu().numpy(), magnitude_error_field=mags[0].cpu().numpy(), 
            vorticity_field=vort_mag.cpu().numpy())

    PSNRs.append(ps)


labels.append("bilinear")
ps = []
mses = []
for ts in range(start_ts, end_ts, ts_skip):
    data = np.load(test_data_folder + str(ts)+".npy")
    data = data[:,:,:,int(data.shape[3]/2)]
    data = np2torch(data, "cuda:0").unsqueeze(0)
    data = dataset.scale(data)

    data_lr = laplace_pyramid_downscale2D(data, 2,
        0.5,
        "cuda:0", opt['periodic'])
    upscaled_data = F.interpolate(data_lr, size=[128,128],
    mode='bilinear', align_corners=False)

    p = PSNR(data, upscaled_data, data.max() - data.min())
    m = ((upscaled_data - data)**2).mean()

    upscaled_data = dataset.unscale(upscaled_data)
    
    ps.append(p)

    if(ts == ts_to_save):
        save_VF(upscaled_data[0], "bilin")
        save_VF(dataset.unscale(data)[0], "gt")

PSNRs.append(ps)

for i in range(len(PSNRs)):
    plt.plot(np.arange(1, (num_ts)*ts_skip+1, ts_skip), PSNRs[i])
plt.legend(labels)
plt.title("Upscaled PSNR over timesteps")
plt.xlabel("Timestep")
plt.ylabel("PSNR")
plt.show()