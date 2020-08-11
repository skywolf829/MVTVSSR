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


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="Temp",help='The type of input - 2D, 3D, or 2D time-varying')
parser.add_argument('--data_folder',default="Isabel/validation",type=str,help='File to train on')
parser.add_argument('--num_testing_examples',default=None,type=int,help='Frames to use from training file')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')
parser.add_argument('--visualize',default=None,type=int,help='channel to visualize via creating a GIF thats saved')

args = vars(parser.parse_args())

dataset = Dataset(os.path.join(input_folder, args["data_folder"]))

frames = []
for i in range(len(dataset)):
    frame = dataset.__getitem__(i)
    frames.append(frame[5].cpu().numpy())

frames = np.array(frames)
frames_min = frames.min()
frames -= frames_min
frames *= (255/frames.max())
frames = frames.astype(np.uint8)
imageio.mimwrite("Precip.gif", frames)