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
parser.add_argument('--data_folder',default="CM1_2/validation",type=str,help='File to train on')
parser.add_argument('--num_testing_examples',default=None,type=int,help='Frames to use from training file')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')
parser.add_argument('--visualize',default=None,type=int,help='channel to visualize via creating a GIF thats saved')

args = vars(parser.parse_args())

opt = load_options(os.path.join(save_folder, args["load_from"]))
opt["device"] = [args["device"]]
opt["save_name"] = args["load_from"]

if(args["num_testing_examples"] is None):
    args["num_testing_examples"] = len(os.listdir(os.path.join(input_folder, args["data_folder"])))
dataset = Dataset(os.path.join(input_folder, args["data_folder"]))
torch.cuda.set_device(args["device"])

psnrs_bilin0 = []
psnrs_bilin1 = []

writer = SummaryWriter(os.path.join('tensorboard', 'validation', 'bilinear'))

for i in range(args["num_testing_examples"]):
    # Get the data
    test_frame = dataset.__getitem__(i).clone()
    y1 = test_frame.clone().cuda().unsqueeze(0)

    # Create the low res versions of it
    lr = F.interpolate(y1.clone(), size=opt["scales"][0], mode=opt["downsample_mode"], align_corners=True)
    y0 = F.interpolate(y1.clone(), size=opt["scales"][1], mode=opt["downsample_mode"], align_corners=True)
    
    # Create bilinear upsampling result
    bilin0 = F.interpolate(lr.clone(), size=opt["scales"][1], mode=opt["upsample_mode"], align_corners=True)
    bilin1 = F.interpolate(lr.clone(), size=opt["scales"][2], mode=opt["upsample_mode"], align_corners=True)

    # Get metrics for the frame
    y0 = y0 + 1
    y1 = y1 + 1
    bilin0 = bilin0 + 1
    bilin1 = bilin1 + 1

    p_bilin0 = psnr(bilin0.clone(), y0.clone(), data_range=2., reduction="none").item()
    p_bilin1 = psnr(bilin1.clone(), y1.clone(), data_range=2., reduction="none").item()

    psnrs_bilin0.append(p_bilin0)
    psnrs_bilin1.append(p_bilin1)

psnrs_bilin0 = np.array(psnrs_bilin0)
psnrs_bilin1 = np.array(psnrs_bilin1)

for i in range(80*400):    
    writer.add_scalar('Validation_PSNR_0', psnrs_bilin0.mean(), i)
    writer.add_scalar('Validation_PSNR_1', psnrs_bilin1.mean(), i)