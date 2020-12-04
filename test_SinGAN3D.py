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
from energy_spectra_analysis import *

def save_VF(vf, name):
    from netCDF4 import Dataset
    rootgrp = Dataset(name+".nc", "w", format="NETCDF4")
    
    u = rootgrp.createDimension("u")
    v = rootgrp.createDimension("v")
    w = rootgrp.createDimension("w")
    c = rootgrp.createDimension("channels", 3)
    us = rootgrp.createVariable("u", np.float32, ("u","v","w"))
    vs = rootgrp.createVariable("v", np.float32, ("u","v","w"))
    ws = rootgrp.createVariable("w", np.float32, ("u","v","w"))
    mags = rootgrp.createVariable("magnitude", np.float32, ("u","v","w"))
    us[:] = vf[0,:,:,:].cpu().numpy()
    vs[:] = vf[1,:,:,:].cpu().numpy()
    ws[:] = vf[2,:,:,:].cpu().numpy()

    m = np.linalg.norm(vf.cpu().numpy(),axis=0)

    mags[:] = m

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="iso128_baseline")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic128_downsampled",
type=str,help='File to test on')
parser.add_argument('--device',default="cuda:0",type=str,
help='Frames to use from training file')

args = vars(parser.parse_args())

opt = load_options(os.path.join(save_folder, args["load_from"]))
opt["device"] = args["device"]


dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)


models_to_try = [
#"iso128_baseline", "iso128_streamline0.5", 
#"iso128_streamline0.5_periodic", 
#"iso128_streamline0.5_periodic2",
#"iso128_streamline0.5_periodic_adaptive",
"iso128_cnn_baseline",
"iso128_cnn_streamlines0.5",
"iso128_cnn_streamlines0.5_adaptive"]

streamline_errors = []
PSNRs = []
MSEs = []
energy_spectra = []
labels = []

for model_name in models_to_try:
    print(model_name)
    labels.append(model_name)
    opt["save_name"] = model_name
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
    energy_spectrum = None
    num_ts = 0
    for ts in range(0, 500, 25):
        data = np.load(str(ts)+".npy")
        data = np2torch(data, "cuda:0").unsqueeze(0)
        data = dataset.scale(data)

        data_lr = laplace_pyramid_downscale3D(data, opt['n']-1,
            opt['spatial_downscale_ratio'],
            "cuda:0")
        
        upscaled_data = generate_by_patch(generators, 
        "reconstruct", opt, 
        opt['device'], opt['patch_size'], 
        generated_image=data_lr, start_scale=1)

        p = PSNR(data, upscaled_data, data.max() - data.min())
        m = ((upscaled_data - data)**2).mean()
        s = streamline_loss3D(data, upscaled_data, 
        opt['streamline_res'], opt['streamline_res'], opt['streamline_res'], 
        1, opt['streamline_length'], opt['device'], periodic=True)

        upscaled_data = dataset.unscale(upscaled_data)
        if(energy_spectrum is None):
            knyquist, wave_numbers, e_s = \
            compute_tke_spectrum(upscaled_data[0,0].cpu().numpy(), 
            upscaled_data[0,1].cpu().numpy(), upscaled_data[0,2].cpu().numpy(), 
            2 * pi, 2 * pi, 2 * pi, True)

            energy_spectrum = np.array(e_s[wave_numbers < knyquist])
        else:
            knyquist, wave_numbers, e_s = \
            compute_tke_spectrum(upscaled_data[0,0].cpu().numpy(), 
            upscaled_data[0,1].cpu().numpy(), upscaled_data[0,2].cpu().numpy(), 
            2 * pi, 2 * pi, 2 * pi, True)
            
            energy_spectrum += np.array(e_s[wave_numbers < knyquist])
        ps.append(p)
        mses.append(m)
        streams.append(s)

        num_ts += 1
        if(ts == 475):
            save_VF(upscaled_data[0], model_name)

    PSNRs.append(ps)
    MSEs.append(mses)
    streamline_errors.append(streams)
    energy_spectra.append(energy_spectrum/num_ts)


labels.append("trilinear")
ps = []
mses = []
streams = []
num_ts = 0
energy_spectrum = None
for ts in range(0, 500, 25):
    data = np.load(str(ts)+".npy")
    data = np2torch(data, "cuda:0").unsqueeze(0)
    data = dataset.scale(data)

    data_lr = laplace_pyramid_downscale3D(data, opt['n']-1,
        opt['spatial_downscale_ratio'],
        "cuda:0")
    
    upscaled_data = F.interpolate(data_lr, size=[128,128,128],
    mode='trilinear', align_corners=True)

    p = PSNR(data, upscaled_data, data.max() - data.min())
    m = ((upscaled_data - data)**2).mean()
    s = streamline_loss3D(data, upscaled_data, 
    opt['streamline_res'], opt['streamline_res'], opt['streamline_res'], 
    1, opt['streamline_length'], opt['device'], periodic=True)

    upscaled_data = dataset.unscale(upscaled_data)
    if(energy_spectrum is None):
        knyquist, wave_numbers, e_s = \
        compute_tke_spectrum(upscaled_data[0,0].cpu().numpy(), 
        upscaled_data[0,1].cpu().numpy(), upscaled_data[0,2].cpu().numpy(), 
        2 * pi, 2 * pi, 2 * pi, True)

        energy_spectrum = np.array(e_s[wave_numbers < knyquist])
    else:
        knyquist, wave_numbers, e_s = \
        compute_tke_spectrum(upscaled_data[0,0].cpu().numpy(), 
        upscaled_data[0,1].cpu().numpy(), upscaled_data[0,2].cpu().numpy(), 
        2 * pi, 2 * pi, 2 * pi, True)
        
        energy_spectrum += np.array(e_s[wave_numbers < knyquist])
    ps.append(p)
    mses.append(m)
    streams.append(s)

    if(ts == 475):
        save_VF(upscaled_data[0], "trilin")
        save_VF(dataset.unscale(data)[0], "gt")

    num_ts += 1
PSNRs.append(ps)
MSEs.append(mses)
streamline_errors.append(streams)
energy_spectra.append(energy_spectrum/num_ts)

for i in range(len(PSNRs)):
    plt.plot(np.arange(1, (num_ts)*25+1, 25), PSNRs[i])
plt.legend(labels)
plt.title("Upscaled PSNR over timesteps")
plt.xlabel("Timestep")
plt.ylabel("PSNR")
plt.show()

for i in range(len(MSEs)):
    plt.plot(np.arange(1, (num_ts)*25+1, 25), MSEs[i])
plt.legend(labels)
plt.title("Upscaled MSE over timesteps")
plt.xlabel("Timestep")
plt.ylabel("MSE")
plt.show()

for i in range(len(streamline_errors)):
    plt.plot(np.arange(1, (num_ts)*25+1, 25), streamline_errors[i])
plt.legend(labels)
plt.title("Upscaled streamline error over timesteps")
plt.xlabel("Timestep")
plt.ylabel("Streamline error")
plt.show()


energy_spectrum = None
for ts in range(0, 500, 25):
    data = np.load(str(ts)+".npy")
    data = np2torch(data, "cuda:0").unsqueeze(0)

    if(energy_spectrum is None):
        knyquist, wave_numbers, e_s = \
        compute_tke_spectrum(data[0,0].cpu().numpy(), 
        data[0,1].cpu().numpy(), data[0,2].cpu().numpy(), 
        2 * pi, 2 * pi, 2 * pi, True)

        energy_spectrum = np.array(e_s[wave_numbers < knyquist])
    
    else:
        knyquist, wave_numbers, e_s = \
        compute_tke_spectrum(data[0,0].cpu().numpy(), 
        data[0,1].cpu().numpy(), data[0,2].cpu().numpy(), 
        2 * pi, 2 * pi, 2 * pi, True)
        
        energy_spectrum += np.array(e_s[wave_numbers < knyquist])
    

energy_spectra.append(energy_spectrum / num_ts)

ks = []
dis=0.0928
for i in wave_numbers[wave_numbers<knyquist]:
    if(i == 0):
        ks.append(1.6 * (dis**(2.0/3.0)))
    else:
        ks.append(1.6 * (dis**(2.0/3.0)) * ((i) **(-5.0/3.0)))
energy_spectra.append(ks)

for i in range(len(energy_spectra)):
    plt.plot(wave_numbers[wave_numbers<knyquist], 
    energy_spectra[i])
plt.legend(labels + ['Ground truth','theoretical dissipation rate'])
plt.title("Energy spectra, averaged over timesteps 1-500")
plt.xlabel("Wavenumber (logscale)")
plt.ylabel("E(k) (logscale)")
plt.yscale("log", nonpositive="clip")
plt.xscale("log", nonpositive="clip")
plt.show()

'''
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
from netCDF4 import Dataset
rootgrp = Dataset("singan.nc", "w", format="NETCDF4")
velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
w = rootgrp.createDimension("channels", 3)
us = rootgrp.createVariable("u", np.float32, ("u","v","w"))
vs = rootgrp.createVariable("v", np.float32, ("u","v","w"))
ws = rootgrp.createVariable("w", np.float32, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", np.float32, ("u","v","w"))
err = rootgrp.createVariable("error", np.float32, ("u","v","w"))
velocities = rootgrp.createVariable("velocities", np.float32, ("u","v","w", "channels"))

singan_output = generate_by_patch(generators, 
"reconstruct", opt, 
opt['device'], opt['patch_size'], 
generated_image=f_lr, start_scale=gen_to_use+1)


us[:] = singan_output[0,0,:,:,:].cpu().numpy()
vs[:] = singan_output[0,1,:,:,:].cpu().numpy()
ws[:] = singan_output[0,2,:,:,:].cpu().numpy()

print(singan_output.shape)
print("SinGAN error:")
e_field = (f_hr - singan_output)**2
e = e_field.mean()
p = PSNR(f_hr, singan_output, f_hr.max() - f_hr.min())
print(e)
print(p)
singan_output = dataset.unscale(singan_output)
singan_output = singan_output.detach().cpu().numpy()[0].swapaxes(0,3).swapaxes(0,1).swapaxes(1,2)
m = np.linalg.norm(singan_output,axis=3)
print(singan_output.shape)
print(m.shape)

mags[:] = m
err[:] = e_field[0].cpu().numpy().sum(axis=0)

















print("Trilinear upscaling:")
rootgrp = Dataset("trilinear.nc", "w", format="NETCDF4")
velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
w = rootgrp.createDimension("channels", 3)
us = rootgrp.createVariable("u", np.float32, ("u","v","w"))
vs = rootgrp.createVariable("v", np.float32, ("u","v","w"))
ws = rootgrp.createVariable("w", np.float32, ("u","v","w"))
err = rootgrp.createVariable("error", np.float32, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", np.float32, ("u","v","w"))
velocities = rootgrp.createVariable("velocities", np.float32, ("u","v","w", "channels"))



print(f_lr.shape)
trilin = F.interpolate(f_lr, 
size=generators[-1].resolution, mode=opt["upsample_mode"], align_corners=True)

us[:] = trilin[0,0,:,:,:].cpu().numpy()
vs[:] = trilin[0,1,:,:,:].cpu().numpy()
ws[:] = trilin[0,2,:,:,:].cpu().numpy()

print(trilin.shape)
e_field = (f_hr - trilin)**2
e = e_field.mean()
p = PSNR(f_hr, trilin, f_hr.max() - f_hr.min())
print(e)
print(p)
trilin = dataset.unscale(trilin)
trilin = trilin.detach().cpu().numpy()[0].swapaxes(0,3).swapaxes(0,1).swapaxes(1,2)
m = np.linalg.norm(trilin,axis=3)
print(trilin.shape)
print(m.shape)

mags[:] = m
err[:] = e_field[0].cpu().numpy().sum(axis=0)


'''