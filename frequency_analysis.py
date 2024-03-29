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


def get_sphere(v, radius, shell_size):
    xxx = torch.arange(0, v.shape[0], dtype=v.dtype, device=v.device).view(-1, 1,1).repeat(1, v.shape[1], v.shape[2])
    yyy = torch.arange(0, v.shape[1], dtype=v.dtype, device=v.device).view(1, -1,1).repeat(v.shape[0], 1, v.shape[2])
    zzz = torch.arange(0, v.shape[2], dtype=v.dtype, device=v.device).view(1, 1,-1).repeat(v.shape[0], v.shape[1], 1)
    sphere = (((xxx-int(v.shape[0]/2))**2) + ((yyy-int(v.shape[1]/2))**2) + ((zzz-int(v.shape[2]/2))**2))**0.5
    sphere = torch.logical_and(sphere < (radius + int(shell_size/2)), sphere > (radius - int(shell_size/2)))
    return sphere

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

dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)
dataset2 = Dataset(os.path.join(input_folder, args["data_folder_diffsize"]), opt)

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

f_trilin = dataset.unscale(F.interpolate(f_lr, mode='trilinear', size=[128, 128, 128])).cpu()[0].numpy()
b, c, d = compute_tke_spectrum(f_trilin[0], f_trilin[1], f_trilin[2], 
2 * pi * (f_trilin.shape[1] / 1024.0), 2 * pi * (f_trilin.shape[2] / 1024), 
2 * pi * (f_trilin.shape[3] / 1024.0), True)

plt.plot(c[c < b], d[c < b], color='blue')


singan_output = generate_by_patch(generators[0:2], 
"reconstruct", opt, 
opt['device'], opt['patch_size'], 
generated_image=f_lr, start_scale=1)
#print(str(i)+": " + str(singan_output.shape))
singan_output = F.interpolate(singan_output, mode='trilinear', size=[128, 128, 128])
singan_output = dataset.unscale(singan_output).cpu().numpy()[0]

b, c, d = compute_tke_spectrum(singan_output[0], singan_output[1], singan_output[2], 
2 * pi * (singan_output.shape[1] / 1024.0), 2 * pi * (singan_output.shape[2] / 1024), 
2 * pi * (singan_output.shape[3] / 1024.0), True)

plt.plot(c[c < b], d[c < b], color='green')

singan_output = generate_by_patch(generators[:], 
"reconstruct", opt, 
opt['device'], opt['patch_size'], 
generated_image=f_lr, start_scale=1)
singan_output = dataset.unscale(singan_output.cpu()).cpu().numpy()[0]

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

'''
f_lr = laplace_pyramid_downscale3D(f, opt['n']-gen_to_use-1,
opt['spatial_downscale_ratio'],
opt['device'])
print(f_lr.shape)
f_singan_fft = []
labels = []
for i in range(len(generators)-1):
    singan_output = generate_by_patch(generators[0:i+2], 
    "reconstruct", opt, 
    opt['device'], opt['patch_size'], 
    generated_image=f_lr, start_scale=1)
    #print(str(i)+": " + str(singan_output.shape))
    singan_output = F.interpolate(singan_output, mode='trilinear', size=[128, 128, 128])
    singan_output = singan_output.cpu().numpy()[0]
    singan_output = np.linalg.norm(singan_output, axis=0)
    f_singan_fft.append(np2torch(np.log(np.abs(np.fft.fftshift(np.fft.fftn(singan_output)))**2), "cuda:0"))
    labels.append("SinGAN - after gen " + str(i+1))

f_trilin = F.interpolate(f_lr, mode='trilinear', size=[128, 128, 128])
f_trilin = f_trilin.cpu().numpy()[0]
print(f_trilin.shape)
labels.append("Trilinear")
f = np.linalg.norm(f[0].cpu().numpy(), axis=0)
f_trilin = np.linalg.norm(f_trilin, axis=0)
f_fft = np.fft.fftn(f)
f_trilin_fft = np.fft.fftn(f_trilin)
labels.append("Ground truth")

'''
'''
f_fft = np.log(np.abs(np.fft.fftshift(f_fft))**2)
plt.imshow(f_fft[256])
plt.show()
'''
'''
f_fft = np2torch(np.log(np.abs(np.fft.fftshift(f_fft))**2), "cuda:0")
f_trilin_fft = np2torch(np.log(np.abs(np.fft.fftshift(f_trilin_fft))**2), "cuda:0")

num_bins = int(f_fft.shape[0] / 2)
#num_bins = 2
bins = []
trilin_bins = []
singan_bins = []
for j in range(len(f_singan_fft)):
    singan_bins.append([])
gif = []
#imageio.imsave("f_fft.png", f_fft[256].cpu().numpy())
for i in range(0, num_bins):
    radius = i * (f_fft.shape[0] / num_bins)
    shell = 5    
    sphere = get_sphere(f_fft, radius, shell)
    s = f_fft * sphere
    gif.append(s[64].cpu().numpy())
    s_trilin = f_trilin_fft * sphere
    on_pixels = sphere.sum().item()
    bins.append(s.sum().item() / (on_pixels+1))
    trilin_bins.append(s_trilin.sum().item() / (on_pixels+1))
    for j in range(len(f_singan_fft)):
        s_singan = f_singan_fft[j] * sphere
        singan_bins[j].append(s_singan.sum().item() / (on_pixels+1))

#imageio.mimwrite("spheres.gif", gif)
ks = []
dis=0.0928
for i in range(num_bins):
    if(i == 0):
        ks.append(1.6 * (dis**(2.0/3.0)))
    else:
        ks.append(1.6 * (dis**(2.0/3.0)) * ((i) **(-5.0/3.0)))
        
for j in range(len(f_singan_fft)):
    plt.plot(np.arange(0, num_bins), singan_bins[j])
plt.plot(np.arange(0, num_bins), trilin_bins)
plt.plot(np.arange(0, num_bins), bins)
#plt.plot(np.arange(0, num_bins), ks, linestyle="dashed", color="black")

plt.yscale("log")
plt.xscale("log")
plt.xlabel("wavenumber")
plt.ylabel("average FFT coeff")
plt.title("FFT coefficients at different wavenumbers for 128^3 volume")
plt.legend(labels)# '1.6*e^(2/3)*k^(-5/3)'])
plt.show()
'''