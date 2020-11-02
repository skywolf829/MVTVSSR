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
    discriminators[i].to("cpu")
    discriminators[i].eval()

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
def get_sphere(v, radius, shell_size):
    xxx = torch.arange(0, v.shape[0], dtype=v.dtype, device=v.device).view(-1, 1,1).repeat(1, v.shape[1], v.shape[2])
    yyy = torch.arange(0, v.shape[1], dtype=v.dtype, device=v.device).view(1, -1,1).repeat(v.shape[0], 1, v.shape[2])
    zzz = torch.arange(0, v.shape[2], dtype=v.dtype, device=v.device).view(1, 1,-1).repeat(v.shape[0], v.shape[1], 1)
    xxx -= int(v.shape[0]/2)
    xxx *= xxx
    yyy -= int(v.shape[1]/2)
    yyy *= yyy
    zzz -= int(v.shape[2]/2)
    zzz *= zzz

    xxx += yyy
    xxx += zzz
    #xxx **= 0.5
    xxx = torch.pow(xxx, 0.5)
    
    xxx = torch.logical_and(xxx < (radius + int(shell_size/2)), xxx > (radius - int(shell_size/2)))
    return xxx

dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)
dataset_diffsize = Dataset(os.path.join(input_folder, args["data_folder_diffsize"]), opt)
gen_to_use = 0

f = dataset.scale(dataset_diffsize.unscale(dataset_diffsize.__getitem__(0).to(opt['device'])))

f_hr = f.to(opt['device'])
del f
f_lr = laplace_pyramid_downscale3D(f_hr, opt['n']-gen_to_use-1,
opt['spatial_downscale_ratio'],
opt['device'])
singan_output = f_lr.clone()
print(f_lr.shape)
print(len(generators))

f_singan_fft = []
labels = []
with torch.no_grad():
    singan_output = F.interpolate(singan_output, size=[256, 256, 256], mode='trilinear', align_corners=True)
    singan_output = generate_patchwise(generators[1], singan_output,
    "reconstruct")
    #print(str(i)+": " + str(singan_output.shape))
    singan_output_np = F.interpolate(singan_output, size=[512, 512, 512], mode='trilinear', align_corners=True).cpu().numpy()[0]
    singan_output_np = np.linalg.norm(singan_output_np, axis=0)
    f_singan_fft.append(np2torch(np.log(np.abs(np.fft.fftshift(np.fft.fftn(singan_output_np)))**2), "cuda:0"))
    labels.append("SinGAN - after gen " + str(1))

    singan_output = F.interpolate(singan_output, size=[512, 512, 512], mode='trilinear', align_corners=True)
    singan_output = generate_patchwise(generators[2], singan_output,
    "reconstruct")
    #print(str(i)+": " + str(singan_output.shape))
    singan_output_np = singan_output.cpu().numpy()[0]
    singan_output_np = np.linalg.norm(singan_output_np, axis=0)
    f_singan_fft.append(np2torch(np.log(np.abs(np.fft.fftshift(np.fft.fftn(singan_output_np)))**2), "cuda:0"))
    labels.append("SinGAN - after gen " + str(2))



f_trilin = F.interpolate(f_lr, mode='trilinear', size=[512, 512, 512])
f_trilin = f_trilin.cpu().numpy()[0]
print(f_trilin.shape)
labels.append("Trilinear")
f = np.linalg.norm(f_hr[0].cpu().numpy(), axis=0)
f_trilin = np.linalg.norm(f_trilin, axis=0)
f_fft = np.fft.fftn(f)
f_trilin_fft = np.fft.fftn(f_trilin)
labels.append("Ground truth")

'''
f_fft = np.log(np.abs(np.fft.fftshift(f_fft))**2)
plt.imshow(f_fft[256])
plt.show()
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
with torch.no_grad():
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
        del s_singan, s_trilin, s, sphere

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
