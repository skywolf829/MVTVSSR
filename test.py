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

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="Temp",help='The type of input - 2D, 3D, or 2D time-varying')
parser.add_argument('--data_folder',default="CM1/test",type=str,help='File to train on')
parser.add_argument('--num_testing_examples',default=201,type=int,help='Frames to use from training file')
parser.add_argument('--device',default="cuda:0",type=str,help='Frames to use from training file')
parser.add_argument('--visualize',default=None,type=int,help='channel to visualize via creating a GIF thats saved')

args = vars(parser.parse_args())

opt = load_options(os.path.join(save_folder, args["load_from"]))
opt["device"] = [args["device"]]
generators, discriminators_s, discriminators_t = load_models(opt,args["device"])

for i in range(len(generators)):
    generators[i].to(args["device"])
    generators[i].eval()
for i in range(len(discriminators_s)):
    discriminators_s[i].to(args["device"])
    discriminators_s[i].eval()
for i in range(len(discriminators_t)):
    discriminators_t[i].to(args["device"])
    discriminators_t[i].eval()

dataset = Dataset(os.path.join(input_folder, args["data_folder"]), args["num_testing_examples"])
torch.cuda.set_device(args["device"])

psnrs = []
ssims = []
rec_frames = []
real_frames = []
print(dataset.channel_mins)
print(dataset.channel_maxs)
for i in range(args["num_testing_examples"]):
    test_frame = dataset.__getitem__(i)
    y = test_frame.cuda().unsqueeze(0)    
    lr = F.interpolate(y.clone(), size=opt["scales"][0], mode=opt["downsample_mode"], align_corners=True)
    bilin = F.interpolate(lr, size=opt["scales"][-1], mode=opt["upsample_mode"], align_corners=True)
    if(args["visualize"] is not None):
        real_frames.append(((y.clone().cpu().numpy()[0, args["visualize"],:,:] + 1) * (255/2)).astype(np.uint8))
    x = F.interpolate(y.clone(), size=opt["scales"][0], mode=opt["downsample_mode"], align_corners=True)
    x = F.interpolate(x, size=opt["scales"][1], mode=opt["upsample_mode"], align_corners=True)
    x = generate(generators, opt, 1, "reconstruct", x, args["device"]).detach()
    x = x.clamp(min=-1, max=1)
    if(args["visualize"] is not None):
        rec_frames.append(((x.clone().cpu().numpy()[0, args["visualize"],:,:] + 1) * (255/2)).astype(np.uint8))

    y = y + 1
    x = x + 1
    bilin = bilin + 1
    p = psnr(x, y, data_range=2., reduction="none").item()
    p_bilin = psnr(bilin, y, data_range=2., reduction="none").item()
    s: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = ssim(x, y, data_range=2.).item()


    print("Frame %i: PSNR: %.04f PSNR_bilin %.04f" % (i, p, p_bilin))

    psnrs.append(p)
    ssims.append(s)

if(args["visualize"] is not None):
    imageio.mimwrite(str(args["visualize"])+".gif", rec_frames)
    imageio.mimwrite("real_"+str(args["visualize"])+".gif", real_frames)
#psnrs = np.array(psnrs)
#ssims = np.array(ssims)

#print(psnrs.mean())
#print(ssims.mean())






