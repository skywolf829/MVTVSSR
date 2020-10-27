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

dataset_diffsize = Dataset(os.path.join(input_folder, args["data_folder_diffsize"]), opt)

gen_to_use = 0

f = dataset.scale(dataset_diffsize.unscale(dataset_diffsize.__getitem__(0).to(opt['device'])))

f_hr = f.to(opt['device'])
del f
f_lr = laplace_pyramid_downscale3D(f_hr, opt['n']-gen_to_use-1,
opt['spatial_downscale_ratio'],
opt['device'])
del f_hr

print(f_lr.shape)
print(len(generators))

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
                    noise = generator.optimal_noise[:,:,z:z_stop,y:y_stop,x:x_stop]
                elif(mode == "random"):
                    noise = torch.randn([generated_image.shape[0], generated_image.shape[1],
                    z_stop-z,y_stop-y,x_stop-x], device=opt['device'])
                
                #print("[%i:%i, %i:%i, %i:%i]" % (z, z_stop, y, y_stop, x, x_stop))
                result = generator(LR[:,:,z:z_stop,y:y_stop,x:x_stop], 
                opt["noise_amplitudes"][i]*noise)

                x_offset = rf if x > 0 else 0
                y_offset = rf if y > 0 else 0
                z_offset = rf if z > 0 else 0

                generated_image[:,:,
                z+z_offset:z+noise.shape[2],
                y+y_offset:y+noise.shape[3],
                x+x_offset:x+noise.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]
    return generated_image

f_lr = F.interpolate(f_lr, size=[256, 256, 256], mode='trilinear')
f_lr = generate_patchwise(generators[1], f_lr, "random")


print(f_lr.shape)
f_lr = F.interpolate(f_lr, size=[512, 512, 512], mode='trilinear')
f_lr = generate_patchwise(generators[2], f_lr, "random")

print(f_lr.shape)