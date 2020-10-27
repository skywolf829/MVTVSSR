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


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="128_SN_0.5")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic128_3D",type=str,help='File to test on')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')

args = vars(parser.parse_args())
opt = Options.get_default()

opt = load_options(os.path.join(save_folder, args["load_from"]))
opt["device"] = args["device"]
opt["save_name"] = args["load_from"]
generators, discriminators = load_models(opt,args["device"])
gen_to_use = 0
for i in range(len(generators)):
    generators[i] = generators[i].to(args["device"])
    generators[i] = generators[i].eval()
for i in range(len(discriminators)):
    discriminators[i].to(args["device"])
    discriminators[i].eval()


def get_sphere(v, radius, shell_size):
    xxx = torch.arange(0, v.shape[0], dtype=v.dtype, device=v.device).view(-1, 1,1).repeat(1, v.shape[1], v.shape[2])
    yyy = torch.arange(0, v.shape[1], dtype=v.dtype, device=v.device).view(1, -1,1).repeat(v.shape[0], 1, v.shape[2])
    zzz = torch.arange(0, v.shape[2], dtype=v.dtype, device=v.device).view(1, 1,-1).repeat(v.shape[0], v.shape[1], 1)
    sphere = (((xxx-int(v.shape[0]/2))**2) + ((yyy-int(v.shape[1]/2))**2) + ((zzz-int(v.shape[2]/2))**2))**0.5
    sphere = torch.logical_and(sphere < (radius + int(shell_size/2)), sphere > (radius - int(shell_size/2)))
    return sphere

dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)

f = np.load(os.path.join(MVTVSSR_folder_path, "InputData", "JHUturbulence", "isotropic128_3D", "0.npy"))
f_torch = np2torch(f, "cuda")
f_lr = laplace_pyramid_downscale3D(f_torch.unsqueeze(0), 2, 0.5, "cuda")
print(f_lr.shape)
singan_output = generate_by_patch(generators, 
"reconstruct", opt, 
opt['device'], opt['patch_size'], 
generated_image=f_lr, start_scale=gen_to_use+1)
singan_output = singan_output.cpu().numpy()[0]
singan_output = np.linalg.norm(singan_output, axis=0)
f_singan_fft = np.fft.fftn(singan_output)
f_trilin = F.interpolate(f_lr, mode='trilinear', size=[128, 128, 128])
f_trilin = f_trilin.cpu().numpy()[0]
print(f_trilin.shape)
f = np.linalg.norm(f, axis=0)
f_trilin = np.linalg.norm(f_trilin, axis=0)
f_fft = np.fft.fftn(f)
f_trilin_fft = np.fft.fftn(f_trilin)

'''
f_fft = np.log(np.abs(np.fft.fftshift(f_fft))**2)
plt.imshow(f_fft[256])
plt.show()
'''
f_fft = np2torch(np.log(np.abs(np.fft.fftshift(f_fft))**2), "cuda:0")
f_trilin_fft = np2torch(np.log(np.abs(np.fft.fftshift(f_trilin_fft))**2), "cuda:0")
f_singan_fft = np2torch(np.log(np.abs(np.fft.fftshift(f_singan_fft))**2), "cuda:0")
num_bins = int(f_fft.shape[0] / 2)
#num_bins = 2
bins = []
trilin_bins = []
singan_bins = []
gif = []
#imageio.imsave("f_fft.png", f_fft[256].cpu().numpy())
for i in range(0, num_bins):
    radius = i * (f_fft.shape[0] / num_bins)
    shell = 5    
    sphere = get_sphere(f_fft, radius, shell)
    s = f_fft * sphere
    gif.append(s[64].cpu().numpy())
    s_trilin = f_trilin_fft * sphere
    s_singan = f_singan_fft * sphere
    on_pixels = sphere.sum().item()
    bins.append(s.sum().item() / (on_pixels+1))
    trilin_bins.append(s_trilin.sum().item() / (on_pixels+1))
    singan_bins.append(s_singan.sum().item() / (on_pixels+1))
imageio.mimwrite("spheres.gif", gif)
ks = []
dis=0.0928
for i in range(num_bins):
    if(i == 0):
        ks.append(1.6 * (dis**(2.0/3.0)))
    else:
        ks.append(1.6 * (dis**(2.0/3.0)) * ((i) **(-5.0/3.0)))
plt.plot(np.arange(0, num_bins), bins)
plt.plot(np.arange(0, num_bins), trilin_bins)
plt.plot(np.arange(0, num_bins), singan_bins)
#plt.plot(np.arange(0, num_bins), ks, linestyle="dashed", color="black")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("wavenumber")
plt.ylabel("average FFT coeff")
plt.title("FFT coefficients at different wavenumbers for 128^3 volume")
plt.legend(["Ground truth", "trilinear, upscaled from 32^3", "singan from 32^3"])# '1.6*e^(2/3)*k^(-5/3)'])
plt.show()
