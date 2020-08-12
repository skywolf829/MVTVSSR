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
generators, discriminators_s, discriminators_t = load_models(opt,args["device"])

for i in range(len(generators)):
    generators[i] = generators[i].to(args["device"])
    generators[i] = generators[i].eval()
for i in range(len(discriminators_s)):
    discriminators_s[i].to(args["device"])
    discriminators_s[i].eval()
for i in range(len(discriminators_t)):
    discriminators_t[i].to(args["device"])
    discriminators_t[i].eval()

if(args["num_testing_examples"] is None):
    args["num_testing_examples"] = len(os.listdir(os.path.join(input_folder, args["data_folder"])))
dataset = Dataset(os.path.join(input_folder, args["data_folder"]))
torch.cuda.set_device(args["device"])

psnrs = []
psnrs_bilin = []
psnr_diff_per_channel = []

first_order_err = []
first_order_err_bilin = []

second_order_err = []
second_order_err_bilin = []

ssims = []
ssims_bilin = []

rec_frames = []
real_frames = []
diffs = []
bilins = []
masks = []

first_orders = []
second_orders = []

rolling_mask = None

writer = SummaryWriter(os.path.join('tensorboard', 'testing',
    "%ikernels_%ilayers_%iminibatch_%iepochs_%.02fdownscaleratio" % (opt["base_num_kernels"], opt["num_blocks"], opt["minibatch"], opt["epochs"], opt["downscale_ratio"])))

with torch.no_grad():
    for i in range(args["num_testing_examples"]):
        # Get the data
        test_frame = dataset.__getitem__(i).clone()
        y = test_frame.clone().cuda().unsqueeze(0)

        # Create the low res version of it
        lr = F.interpolate(y.clone(), size=opt["scales"][0], mode=opt["downsample_mode"])

        # Create bilinear upsampling result
        bilin = F.interpolate(lr.clone(), size=opt["scales"][-1], mode=opt["upsample_mode"])
        
        # Create network reconstructed result
        x = F.interpolate(y.clone(), size=opt["scales"][0], mode=opt["downsample_mode"])
        x = F.interpolate(x, size=opt["scales"][1], mode=opt["upsample_mode"])
        x = generate(generators, opt, 1, "reconstruct", x, args["device"]).detach()
        x = x.clamp(min=-1, max=1)
        # Calculate a frame difference for masking
        if i == 0:
            next_frame = dataset.__getitem__(i+1).clone().cuda().unsqueeze(0)
            next_frame = F.interpolate(next_frame, size=opt["scales"][0], mode=opt["downsample_mode"])
            frame_diff = torch.abs(next_frame - lr.clone())

        else: 
            last_frame = dataset.__getitem__(i-1).clone().cuda().unsqueeze(0)
            last_frame = F.interpolate(last_frame, size=opt["scales"][0], mode=opt["downsample_mode"])
            frame_diff = torch.abs(last_frame - lr.clone())
        frame_diff = F.interpolate(frame_diff, size=opt["scales"][-1], mode=opt["upsample_mode"])

        if rolling_mask is None:
            rolling_mask = torch.zeros(frame_diff.shape).cuda()
        mask = torch.zeros(frame_diff.shape).cuda()
        mask_indices = torch.where(frame_diff > 1e-8)
        mask[mask_indices] = 1.0

        # Apply blur to the mask
        gaussian = GaussianSmoothing(1, 21, 3).cuda()
        for j in range(mask.shape[1]):
            mask[0:1,j:j+1] = gaussian(F.pad(mask[0:1,j:j+1], (10, 10, 10, 10), mode='reflect'))

        rolling_mask += mask
        rolling_mask = rolling_mask.clamp(0, 1)

        x = (1-rolling_mask.clone()) * bilin.clone() + (rolling_mask.clone()) * x.clone()
        x = x.clamp(min=-1, max=1)

        # Calculate gradients
        s = list(x.shape)
        s[1] = s[1] * 2
        first_order_errs = np.zeros(s)
        first_order_errs_bilin = np.zeros(s)
        s[1] = s[1] * 2
        second_order_errs = np.zeros(s)
        second_order_errs_bilin = np.zeros(s)

        for j in range(y.shape[1]):
            x_np = x.clone().cpu().numpy()[0,j]
            y_np = y.clone().cpu().numpy()[0,j]
            bilin_np = bilin.clone().cpu().numpy()[0,j]

            # First orders
            x_grads_y, x_grads_x = np.gradient(x_np)
            y_grads_y, y_grads_x = np.gradient(y_np)
            bilin_grads_x, bilin_grads_y = np.gradient(bilin_np)
            
            first_order_errs[0,j] = (x_grads_y - y_grads_y) ** 2
            first_order_errs[0,j+1] = (x_grads_x - y_grads_x) ** 2
            first_order_errs_bilin[0,j] = (bilin_grads_y - y_grads_y) ** 2
            first_order_errs_bilin[0,j+1] = (bilin_grads_x - y_grads_x) ** 2

            # Second orders
            x_d2_dy2, x_d2_dydx = np.gradient(x_grads_y)
            x_d2_dxdy, x_d2_dx2 = np.gradient(x_grads_x)

            y_d2_dy2, y_d2_dydx = np.gradient(y_grads_y)
            y_d2_dxdy, y_d2_dx2 = np.gradient(y_grads_x)

            bilin_d2_dy2, bilin_d2_dydx = np.gradient(bilin_grads_y)
            bilin_d2_dxdy, bilin_d2_dx2 = np.gradient(bilin_grads_x)

            second_order_errs[0,j] = (y_d2_dy2 - x_d2_dy2) ** 2        
            second_order_errs[0,j+1] = (y_d2_dydx - x_d2_dydx) ** 2
            second_order_errs[0,j+2] = (y_d2_dxdy - x_d2_dxdy) ** 2
            second_order_errs[0,j+3] = (y_d2_dx2 - x_d2_dx2) ** 2

            second_order_errs_bilin[0,j] = (y_d2_dy2 - bilin_d2_dy2) ** 2        
            second_order_errs_bilin[0,j+1] = (y_d2_dydx - bilin_d2_dydx) ** 2
            second_order_errs_bilin[0,j+2] = (y_d2_dxdy - bilin_d2_dxdy) ** 2
            second_order_errs_bilin[0,j+3] = (y_d2_dx2 - bilin_d2_dx2) ** 2

        first_order_err.append(np.mean(first_order_errs))
        first_order_err_bilin.append(np.mean(first_order_errs_bilin))

        second_order_err.append(np.mean(second_order_errs))
        second_order_err_bilin.append(np.mean(second_order_errs_bilin))
        

        if(args["visualize"] is not None):
            rec_frames.append(((x.clone().cpu().numpy()[0, args["visualize"],:,:] + 1) * (255/2)).astype(np.uint8))        
            real_frames.append(((y.clone().cpu().numpy()[0, args["visualize"],:,:] + 1) * (255/2)).astype(np.uint8))
            masks.append(((mask.clone().cpu().numpy()[0, args["visualize"],:,:]) * (255)).astype(np.uint8))
            diff = torch.abs(y.clone().detach() - x.clone().detach())
            diffs.append((((diff.cpu().numpy()[0, args["visualize"],:,:] / 2) ** 0.35) * (255/1)).astype(np.uint8))
            bilins.append(((bilin.clone().cpu().numpy()[0, args["visualize"],:,:]+1) * (255/2)).astype(np.uint8))

        # Get metrics for the frame
        y = y + 1
        x = x + 1
        bilin = bilin + 1

        for j in range(x.shape[1]):
            p1 = psnr(x.clone()[:,j:j+1,:,:], y.clone()[:,j:j+1,:,:], data_range=2., reduction="none").item()
            p2 = psnr(bilin.clone()[:,j:j+1,:,:], y.clone()[:,j:j+1,:,:], data_range=2., reduction="none").item()
            pdiff = p1-p2
            if(len(psnr_diff_per_channel) <= j):
                psnr_diff_per_channel.append([])
            psnr_diff_per_channel[j].append(pdiff)

        p = psnr(x.clone(), y.clone(), data_range=2., reduction="none").item()
        p_bilin = psnr(bilin.clone(), y.clone(), data_range=2., reduction="none").item()
        s: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = ssim(x.clone(), y.clone(), data_range=2.).item()
        s_bilin: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = ssim(bilin.clone(), y.clone(), data_range=2.).item()

        writer.add_scalar('PSNR', p, i)
        writer.add_scalar('SSIM', s, i)
        writer.add_scalar('First_order_MSE', np.mean(first_order_errs), i)
        writer.add_scalar('Second_order_MSE', np.mean(second_order_errs), i)

        print("Frame %i: PSNR: %.04f PSNR_bilin %.04f" % (i, p, p_bilin))
        
        psnrs.append(p)
        psnrs_bilin.append(p_bilin)
        ssims.append(s)
        ssims_bilin.append(s_bilin)

if(args["visualize"] is not None):
    imageio.mimwrite(str(args["visualize"])+".gif", rec_frames)
    imageio.mimwrite("real_"+str(args["visualize"])+".gif", real_frames)
    imageio.mimwrite("diff_"+str(args["visualize"])+".gif", diffs)
    imageio.mimwrite("bilin_"+str(args["visualize"])+".gif", bilins)
    imageio.mimwrite("mask_"+str(args["visualize"])+".gif", masks)
    #imageio.mimwrite("1storder_"+str(args["visualize"])+".gif", first_orders)
    #imageio.mimwrite("2ndorder_"+str(args["visualize"])+".gif", second_orders)

psnrs = np.array(psnrs)
psnrs_bilin = np.array(psnrs_bilin)
ssims = np.array(ssims)
ssims_bilin = np.array(ssims_bilin)
first_order_err = np.array(first_order_err)
first_order_err_bilin = np.array(first_order_err_bilin)
second_order_err = np.array(second_order_err)
second_order_err_bilin = np.array(second_order_err_bilin)

print("Mean PSNR: %.03f" % psnrs.mean())
print("Mean SSIM: %.03f" % ssims.mean())
print("Mean bilinear PSNR: %.03f" % psnrs_bilin.mean())
print("Mean bilinear SSIM: %.03f" % ssims_bilin.mean())

create_graph(np.arange(len(psnrs)), [psnrs, psnrs_bilin], 
"PSNR per frame on 17 channel CM1 data", 
"PSNR", "frame", ['blue', 'red'], ['Network', 'Bilinear'])

create_graph(np.arange(len(psnrs)), psnr_diff_per_channel, 
"PSNR difference (network - bilinear) per channel", 
"PSNR", "frame", cm.tab20(np.linspace(0,1,len(psnr_diff_per_channel))), [str(i) for i in range(len(psnr_diff_per_channel))])

create_graph(np.arange(len(psnrs)), [ssims, ssims_bilin], 
"SSIM per frame on 17 channel CM1 data", 
"SSIM", "frame", ['blue', 'red'], ['Network', 'Bilinear'])

create_graph(np.arange(len(psnrs)), [first_order_err, first_order_err_bilin], 
"First order gradient error per frame on 17 channel CM1 data", 
"First order gradient error", "frame", ['blue', 'red'], ['Network', 'Bilinear'])

create_graph(np.arange(len(psnrs)), [second_order_err, second_order_err_bilin], 
"Second order error per frame on 17 channel CM1 data", 
"Second order error", "frame", ['blue', 'red'], ['Network', 'Bilinear'])

if(args['visualize'] is not None):
    mins = []
    maxs = []
    medians = []
    for i in range(args["num_testing_examples"]):
        test_frame = dataset.__getitem__(i).numpy()
        mins.append(test_frame[args['visualize']].min())
        maxs.append(test_frame[args['visualize']].max())
        medians.append(np.median(test_frame[args['visualize']]))

    create_graph(np.arange(len(maxs)),  [maxs, mins, medians], 
    "Min/max of channel %i per frame on 17 channel CM1 data" % args['visualize'], 
    "value", "frame", ['red', 'blue', 'green'], ['Max', 'Min', 'Median'])



'''
Stats:

bilinear:
45.957 PSNR
0.99 SSIM

1 scale CNN 32 kernels 15 epochs minibatch 1:
43.451 PSNR
0.976 SSIM

2 scale CNN 32 kernels 15 epochs minibatch 1:
43.298 PSNR
0.979 SSIM

3 scale CNN 32 kernels 15 epochs minibatch 1:
42.523 PSNR
0.983 SSIM


====================================================

2 scale GAN 64 kernels 15 epochs minibatch 1:
41.255 PSNR
0.983 SSIM

'''