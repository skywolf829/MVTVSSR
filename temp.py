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




MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")


a = np.load("0.npy")
a = np2torch(a, "cuda:1").unsqueeze(0)
b = torch.zeros(a.shape)
b= laplace_pyramid_downscale3D(a[:,0:1], 2, 0.5, "cuda:1")
c= laplace_pyramid_downscale3D(a[:,1:2], 2, 0.5, "cuda:1")
d= laplace_pyramid_downscale3D(a[:,2:3], 2, 0.5, "cuda:1")
e = torch.concat([b,c,d], axis=1)
print(e.shape)