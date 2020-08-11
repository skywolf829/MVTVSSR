from model import *
from options import *
from utility_functions import *
import torch.nn.functional as F
import torch
from piq import ssim, psnr, multi_scale_ssim
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

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="Temp",help='The type of input - 2D, 3D, or 2D time-varying')
parser.add_argument('--data_folder',default="CM1_2/validation",type=str,help='File to train on')
parser.add_argument('--num_testing_examples',default=None,type=int,help='Frames to use from training file')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')
parser.add_argument('--visualize',default=None,type=int,help='channel to visualize via creating a GIF thats saved')

args = vars(parser.parse_args())

torch.cuda.set_device(args["device"])

entropys = []
dataset = Dataset(os.path.join(input_folder, args["data_folder"]))

channel = 3

for s in range(15):
    scale = 1.0 - s * 0.05
    ents = []
    for i in range(len(dataset)):
        # Get the data
        test_frame = dataset.__getitem__(i).clone().unsqueeze(0).cuda()
        shape = list(test_frame.shape[2:])
        shape[0] = int(shape[0] * scale)
        shape[1] = int(shape[1] * scale)
        if(s is not 0):
            test_frame = F.interpolate(test_frame, size=shape, mode='bilinear', align_corners=True)
        probs = data_to_bin_probability(test_frame[0,channel], 100)
        #print(probs)
        ent = calculate_entropy(probs)
        #print(ent)
        ents.append(ent)
    print((np.array(ents)).mean())