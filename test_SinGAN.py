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



def display_2d(frame):
    scale = 16
    x = np.meshgrid(np.linspace(-1.0, 1.0, int(frame.shape[1]/scale)))
    y = np.meshgrid(np.linspace(-1.0, 1.0, int(frame.shape[2]/scale)))
    plt.quiver(x, y, frame[1,::scale,::scale], frame[0,::scale,::scale], 
    pivot='middle',linestyle='solid')
    plt.show()

def display_2d_mag(frame):
    shape = list(frame.shape)
    shape[0] = 1
    new_chan = np.zeros(shape)
    print(new_chan.shape)
    new_chan[0] = (frame[1]**2 + frame[0]**2)**0.5
    new_chan = new_chan.swapaxes(0,2).swapaxes(0,1)
    plt.imshow(new_chan, cmap="gray")
    plt.show()

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

parser = argparse.ArgumentParser(description='Test a trained model')

parser.add_argument('--load_from',default="L1_512")
parser.add_argument('--data_folder',default="JHUturbulence/isotropic1024coarse",type=str,help='File to test on')
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

'''
# Zero shot super resolution

dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)

frame = dataset.__getitem__(0).to(opt['device'])
f_LR = frame.clone()
f_mag = torch.norm(frame, dim=1).detach().cpu().numpy()
f_np = frame[0].detach().cpu().numpy()

plt.hist(f_np[0].flatten(), 50, histtype='step', stacked=True, fill=False, color='blue')
plt.hist(f_np[1].flatten(), 50, histtype='step', stacked=True, fill=False, color='green')
plt.hist(f_np[2].flatten(), 50, histtype='step', stacked=True, fill=False, color='orange')
plt.legend(['u', 'v', 'w'])
plt.title('data distribution - GT')
plt.show()
imageio.imwrite("GT_LR_mag.png", toImg(f_mag).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("GT_LR_uvw.png", toImg(f_np).swapaxes(0,2).swapaxes(0,1))

f_singan = generate(generators, "reconstruct", opt, opt["device"])
f_singan_mag = torch.norm(f_singan, dim=1).detach().cpu().numpy()
f_singan = f_singan[0].detach().cpu().numpy()
plt.hist(f_singan[0].flatten(), 50, histtype='step', stacked=True, fill=False, color='blue')
plt.hist(f_singan[1].flatten(), 50, histtype='step', stacked=True, fill=False, color='green')
plt.hist(f_singan[2].flatten(), 50, histtype='step', stacked=True, fill=False, color='orange')
plt.legend(['u', 'v', 'w'])
plt.title('data distribution - %s' % args["load_from"])
plt.show()
imageio.imwrite("LR_singan_mag.png", toImg(f_singan_mag).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("LR_singan_uvw.png", toImg(f_singan).swapaxes(0,2).swapaxes(0,1))

scaling = 4
dataset = Dataset(os.path.join(input_folder, "JHUturbulence", "isotropic512coarse"), opt)
frame = dataset.__getitem__(0).to(opt['device'])
f_mag = torch.norm(frame, dim=1).detach().cpu().numpy()
f_np = frame[0].detach().cpu().numpy()
imageio.imwrite("GT_HR_mag.png", toImg(f_mag).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("GT_HR_uvw.png", toImg(f_np).swapaxes(0,2).swapaxes(0,1))

bicubic = F.interpolate(f_LR, scale_factor=scaling,mode=opt["upsample_mode"])
bicubic_mag = torch.norm(bicubic, dim=1).detach().cpu().numpy()
bicubic_np = bicubic[0].detach().cpu().numpy()
imageio.imwrite("bicub_HR_mag.png", toImg(bicubic_mag).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("bicub_HR_uvw.png", toImg(bicubic_np).swapaxes(0,2).swapaxes(0,1))
total_err = (abs(bicubic - frame)).sum()
mse = ((bicubic - frame)**2).mean()

print("Total err bicubic %0.02f" % total_err)
print("MSE bicubic %0.02f" % mse)
singan = super_resolution(generators[-1], f_LR, scaling, opt, opt['device'])
singan_mag = torch.norm(singan, dim=1).detach().cpu().numpy()
singan_np = singan[0].detach().cpu().numpy()
imageio.imwrite("singan_HR_mag.png", toImg(singan_mag).swapaxes(0,2).swapaxes(0,1))
imageio.imwrite("singan_HR_uvw.png", toImg(singan_np).swapaxes(0,2).swapaxes(0,1))
total_err = (abs(singan - frame)).sum()
mse = ((singan - frame)**2).mean()

print("Total err singan %0.02f" % total_err)
print("MSE singan %0.02f" % mse)
'''


# One shot super resolution
dataset = Dataset(os.path.join(input_folder, args["data_folder"]), opt)


GT_frames = []
gt_mag = []
bicub_frames = []
bicub_mag = []
singan_frames = []
singan_mag = []

bicub_err = []
singan_err = []
print(len(dataset))

gen_to_use = 5
lr = opt['resolutions'][gen_to_use]

for i in range(len(dataset)):
    print(i)
    f = dataset.__getitem__(i)
    f_lr = pyramid_reduce(f[0].clone().cpu().numpy().swapaxes(0,2).swapaxes(0,1), 
        downscale = f.shape[2] / lr[0],
        multichannel=True).swapaxes(0,1).swapaxes(0,2)
    f_lr = np2torch(f_lr, opt['device']).unsqueeze(0).to(opt['device'])
    f_hr = f.clone()[:,:,::2,::2].to(opt['device'])
    g_mag = torch.norm(f_hr, dim=1).detach().cpu().numpy()
    GT_frames.append(toImg(f_hr.clone()[0].detach().cpu().numpy()).swapaxes(0,2).swapaxes(0,1))
    gt_mag.append(toImg(g_mag).swapaxes(0,2).swapaxes(0,1))

    bicub = F.interpolate(f_lr.clone(), size=opt['resolutions'][-1],mode=opt["upsample_mode"])
    b_mag = torch.norm(bicub, dim=1).detach().cpu().numpy()
    bicub_frames.append(toImg(bicub.clone().detach().cpu().numpy()[0]).swapaxes(0,2).swapaxes(0,1))
    bicub_err.append(abs(bicub-f_hr).sum())
    bicub_mag.append(toImg(b_mag).swapaxes(0,2).swapaxes(0,1))


    generated = f_lr.clone()
    for j in range(gen_to_use+1, len(generators)):
        with torch.no_grad():
            generated = F.interpolate(generated, size=opt['resolutions'][j],mode=opt["upsample_mode"]).detach()
            generated = generators[j](generated, 
            torch.randn(generators[j].get_input_shape()).to(opt["device"]) * opt['noise_amplitudes'][j])
            #generators[j].optimal_noise * opt['noise_amplitudes'][j])
    s_mag = torch.norm(generated, dim=1).detach().cpu().numpy()
    singan_frames.append(toImg(generated.detach().cpu().numpy()[0]).swapaxes(0,2).swapaxes(0,1))
    singan_err.append(abs(generated-f_hr).sum())
    singan_mag.append(toImg(s_mag).swapaxes(0,2).swapaxes(0,1))

imageio.mimwrite("singan.gif", singan_frames)
imageio.mimwrite("singan_mag.gif", singan_mag)
imageio.mimwrite("GT.gif", GT_frames)
imageio.mimwrite("gt_mag.gif", gt_mag)
imageio.mimwrite("bicub.gif", bicub_frames)
imageio.mimwrite("bicub_mag.gif", bicub_mag)

imageio.imwrite("singan_last_frame_zoom.png", singan_frames[-1][100:200, 100:200, :])
imageio.imwrite("singan_last_frame_mag_zoom.png", singan_mag[-1][100:200, 100:200, :])
imageio.imwrite("bicub_last_frame_zoom.png", bicub_frames[-1][100:200, 100:200, :])
imageio.imwrite("bicub_last_frame_mag_zoom.png", bicub_mag[-1][100:200, 100:200, :])
imageio.imwrite("gt_last_frame_zoom.png", GT_frames[-1][100:200, 100:200, :])
imageio.imwrite("gt_last_frame_mag_zoom.png", gt_mag[-1][100:200, 100:200, :])


plt.plot(np.arange(0, len(singan_err)), singan_err, color='red')
plt.plot(np.arange(0, len(singan_err)), bicub_err, color='blue')
plt.legend(['SinGAN', 'Bicubic'])
plt.title('Absolute error per frame')
plt.show()
