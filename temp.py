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

def data_to_bin_probability(data, num_bins):
    probs = []
    data = data.clone().flatten()
    d_min = -1
    d_max = 1
    for i in range(num_bins):
        bin_min = d_min + i * ((d_max-d_min) / num_bins)
        bin_max = d_min + (i+1) * ((d_max-d_min) / num_bins)
        if(i == num_bins - 1):            
            indices = torch.where((data >= bin_min) & (data <= bin_max))
        else:
            indices = torch.where((data >= bin_min) & (data < bin_max))
        probs.append(indices[0].shape[0] / data.shape[0])
    return probs

def calculate_entropy(probs):
    ent = 0.
    for i in range(len(probs)):
        ent -= probs[i] * log(probs[i] + 1e-8, e)
    return ent


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

torch.cuda.set_device(0)

'''
dataset = Dataset(os.path.join(input_folder, "Synthetic", "test"))
psnrs = []
for i in range(len(dataset)):
    test_frame = dataset.__getitem__(i)
    y = test_frame.clone().cuda().unsqueeze(0)
    
    # Create the low res version of it
    lr = F.interpolate(y.clone(), size=[32, 32], mode="nearest")
    hr = F.interpolate(lr, size=[128, 128], mode="bicubic")
    p = 20*math.log(2) - 10*math.log(torch.mean((hr - y)**2).item())
    psnrs.append(p)
print((np.array(psnrs)).mean())


opt = load_options(os.path.join(save_folder, "Temp"))
opt["device"] = [0]
opt["save_name"] = "Temp"
generators, discriminators_s = load_models(opt,"cuda:0")

for i in range(len(generators)):
    generators[i] = generators[i].to(0)
    generators[i] = generators[i].eval()
psnrs = []
for i in range(len(dataset)):
    test_frame = dataset.__getitem__(i)
    y = test_frame.clone().cuda().unsqueeze(0)
    
    # Create the low res version of it
    lr = F.interpolate(y.clone(), size=[32, 32], mode="nearest")
    hr = generate(generators, opt, 1, lr, "cuda:0").detach()
    p = 20*math.log(2) - 10*math.log(torch.mean((hr - y)**2).item())
    psnrs.append(p)
print((np.array(psnrs)).mean())

'''


'''
arr = np.zeros([2, 512, 512])
for x in range(arr.shape[1]):
    for y in range(arr.shape[2]):
        arr[0, x, y] = (y-256)/256
        arr[1, x, y] = -(x-256)/256

np.save(os.path.join(input_folder, "Synthetic_VFD", "0_512x512.npy"), arr)
np.save(os.path.join(input_folder, "Synthetic_VFD", "0_256x256.npy"), arr[:,::2,::2])
'''

potential = torch.randn([1, 1, 512, 512], device="cuda:0")
x = spatial_derivative2D(potential, "x", "cuda:0")
y = spatial_derivative2D(potential, "y", "cuda:0")
vf = torch.cat([-y, x], axis=1)
vf = vf.detach().cpu().numpy()[0]
print(vf.shape)
np.save(os.path.join(input_folder, "Synthetic_VFD", "0_512x512.npy"), vf)
np.save(os.path.join(input_folder, "Synthetic_VFD", "0_256x256.npy"), vf[:,::2,::2])