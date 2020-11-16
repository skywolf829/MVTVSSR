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
from scipy import signal
import tidynamics
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="128_GP_0.5")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic128_3D",type=str,help='File to test on')
parser.add_argument('--data_folder_diffsize',default="JHUturbulence/isotropic512_3D",type=str,help='File to test on')
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

def get_ks(x, y, z, xmax, ymax, zmax, device):
    xxx = torch.arange(x,device=device).type(torch.cuda.FloatTensor).view(-1, 1,1).repeat(1, y, z)
    yyy = torch.arange(y,device=device).type(torch.cuda.FloatTensor).view(1, -1,1).repeat(x, 1, z)
    zzz = torch.arange(z,device=device).type(torch.cuda.FloatTensor).view(1, 1,-1).repeat(x, y, 1)
    xxx[xxx>xmax] -= xmax*2
    yyy[yyy>ymax] -= ymax*2
    zzz[zzz>zmax] -= zmax*2
    ks = (xxx*xxx + yyy*yyy + zzz*zzz) ** 0.5
    ks = torch.round(ks).type(torch.LongTensor)
    return ks

def movingaverage(interval, window_size):
    window= ones(int(window_size))/float(window_size)
    return convolve(interval, window, 'same')

def generate_patchwise(generator, LR, mode):
    #print("Gen " + str(i))
    patch_size = 64
    rf = int(generator.receptive_field() / 2)
    generated_image = torch.zeros(LR.shape).to(opt['device'])
    for z in range(0,generated_image.shape[2], patch_size-2*rf):
        z = min(z, max(0, generated_image.shape[2] - patch_size))
        z_stop = min(generated_image.shape[2], z + patch_size)
        for y in range(0,generated_image.shape[3], patch_size-2*rf):
            y = min(y, max(0, generated_image.shape[3] - patch_size))
            y_stop = min(generated_image.shape[3], y + patch_size)

            for x in range(0,generated_image.shape[4], patch_size-2*rf):
                x = min(x, max(0, generated_image.shape[4] - patch_size))
                x_stop = min(generated_image.shape[4], x + patch_size)

                if(mode == "reconstruct"):
                    #noise = generator.optimal_noise[:,:,z:z_stop,y:y_stop,x:x_stop]
                    noise = torch.zeros([1, 3,z_stop-z,y_stop-y,x_stop-x], device="cuda")
                elif(mode == "random"):
                    noise = torch.randn([generated_image.shape[0], generated_image.shape[1],
                    z_stop-z,y_stop-y,x_stop-x], device=opt['device'])
                
                #print("[%i:%i, %i:%i, %i:%i]" % (z, z_stop, y, y_stop, x, x_stop))
                result = generator(LR[:,:,z:z_stop,y:y_stop,x:x_stop], 
                noise)

                x_offset = rf if x > 0 else 0
                y_offset = rf if y > 0 else 0
                z_offset = rf if z > 0 else 0

                generated_image[:,:,
                z+z_offset:z+noise.shape[2],
                y+y_offset:y+noise.shape[3],
                x+x_offset:x+noise.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]
    return generated_image

def compute_tke_spectrum(u,v,w,lx,ly,lz,smooth):
    """
    Given a velocity field u, v, w, this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the 
    following steps:
    1. Compute the spectral representation of u, v, and w using a fast Fourier transform.
    This returns uf, vf, and wf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf, vf, wf)* conjugate(uf, vf, wf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

    Parameters:
    -----------  
    u: 3D array
    The x-velocity component.
    v: 3D array
    The y-velocity component.
    w: 3D array
    The z-velocity component.    
    lx: float
    The domain size in the x-direction.
    ly: float
    The domain size in the y-direction.
    lz: float
    The domain size in the z-direction.
    smooth: boolean
    A boolean to smooth the computed spectrum for nice visualization.
    """
    nx = len(u[:,0,0])
    ny = len(v[0,:,0])
    nz = len(w[0,0,:])

    nt= nx*ny*nz
    n = nx #int(np.round(np.power(nt,1.0/3.0)))

    uh = np.fft.fftn(u)/nt
    vh = np.fft.fftn(v)/nt
    wh = np.fft.fftn(w)/nt

    tkeh = zeros((nx,ny,nz))
    tkeh = 0.5*(uh*conj(uh) + vh*conj(vh) + wh*conj(wh)).real

    k0x = 2.0*pi/lx
    k0y = 2.0*pi/ly
    k0z = 2.0*pi/lz

    knorm = (k0x + k0y + k0z)/3.0

    kxmax = nx/2
    kymax = ny/2
    kzmax = nz/2

    wave_numbers = knorm*arange(0,n)

    tke_spectrum = zeros(len(wave_numbers))
    ks = get_ks(nx, ny, nz, kxmax, kymax, kzmax, "cuda:0")
    tkeh = np2torch(tkeh, "cuda:0")
    for k in range(0, min(len(tke_spectrum), ks.max())):
        print(k)
        tke_spectrum[k] = torch.sum(tkeh[ks == k]).item()
    tkeh = tkeh.cpu().numpy()
    '''
    for kx in range(nx):
        rkx = kx
        if (kx > kxmax):
            rkx = rkx - (nx)
        for ky in range(ny):
            rky = ky
            if (ky>kymax):
                rky=rky - (ny)
            for kz in range(nz):        
                rkz = kz
                if (kz>kzmax):
                    rkz = rkz - (nz)
                rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
                k = int(np.round(rk))
                tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]
    '''
    tke_spectrum = tke_spectrum/knorm
    #  tke_spectrum = tke_spectrum[1:]
    #  wave_numbers = wave_numbers[1:]
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm*min(nx,ny,nz)/2 

    return knyquist, wave_numbers, tke_spectrum

dataset2 = Dataset(os.path.join(input_folder, args["data_folder"]), opt)
dataset = Dataset(os.path.join(input_folder, args["data_folder_diffsize"]), opt)

f = dataset[0].cuda()

a = dataset.unscale(f).cpu().numpy()[0]

b, c, d = compute_tke_spectrum(a[0], a[1], a[2], 
2 * pi * (a.shape[1] / 1024.0), 2 * pi * (a.shape[2] / 1024), 
2 * pi * (a.shape[3] / 1024.0), True)

ks = []
dis=0.0928
for i in c:
    if(i == 0):
        ks.append(1.6 * (dis**(2.0/3.0)))
    else:
        ks.append(1.6 * (dis**(2.0/3.0)) * ((i) **(-5.0/3.0)))
ks = np.array(ks)
c[0] += 1


plt.plot(c[c < b], ks[c < b], color='black')
plt.plot(c[c < b], d[c < b], color='red')

f_lr = laplace_pyramid_downscale3D(f, opt['n']-gen_to_use-1,
opt['spatial_downscale_ratio'],
opt['device'])

f_trilin = dataset.unscale(F.interpolate(f_lr, mode='trilinear', size=[512, 512, 512])).cpu()[0].numpy()
b, c, d = compute_tke_spectrum(f_trilin[0], f_trilin[1], f_trilin[2], 
2 * pi * (f_trilin.shape[1] / 1024.0), 2 * pi * (f_trilin.shape[2] / 1024), 
2 * pi * (f_trilin.shape[3] / 1024.0), True)

plt.plot(c[c < b], d[c < b], color='blue')

with torch.no_grad():
    singan_output = f_lr.clone()
    singan_output = F.interpolate(singan_output, size=[256, 256, 256], mode='trilinear', align_corners=True)
    singan_output = generate_patchwise(generators[1], singan_output,
    "reconstruct")

b, c, d = compute_tke_spectrum(singan_output[0], singan_output[1], singan_output[2], 
2 * pi * (singan_output.shape[1] / 1024.0), 2 * pi * (singan_output.shape[2] / 1024), 
2 * pi * (singan_output.shape[3] / 1024.0), True)

plt.plot(c[c < b], d[c < b], color='green')
with torch.no_grad():
    singan_output = F.interpolate(singan_output, size=[512, 512, 512], mode='trilinear', align_corners=True)
    singan_output = generate_patchwise(generators[2], singan_output,
    "reconstruct")
b, c, d = compute_tke_spectrum(singan_output[0], singan_output[1], singan_output[2], 
2 * pi * (singan_output.shape[1] / 1024.0), 2 * pi * (singan_output.shape[2] / 1024), 
2 * pi * (singan_output.shape[3] / 1024.0), True)

plt.plot(c[c < b], d[c < b], color='orange')

plt.yscale("log", nonpositive="clip")
plt.xscale("log", nonpositive="clip")
plt.xlabel("wavenumber")
plt.title("Energy Spectra")
plt.legend(["1.6*epsilon^(2/3)*k^(-5/3)", "Ground truth data", "Trilinear", 
"SinGAN - intermediate", "SinGAN - final"])
plt.show()
