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
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.transform.pyramids import pyramid_reduce
from energy_spectra_analysis import *
from scipy import stats

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

model_name = "0.5_32_zero"
test_data_folder = "TestingData/iso128/"
ts_to_test = 0

opt = load_options(os.path.join(save_folder, model_name))
opt['device'] = "cuda:0"
generators, discriminators = load_models(opt,opt["device"])
dataset = Dataset(os.path.join(input_folder, "JHUturbulence/isotropic128_downsampled"), opt)

for i in range(len(generators)):
    generators[i] = generators[i].to(opt["device"])
    generators[i] = generators[i].eval()
for i in range(len(discriminators)):
    discriminators[i].to(opt["device"])
    discriminators[i].eval()

data = np.load(test_data_folder + str(ts_to_test)+".npy")
data = data[:,:,:,int(data.shape[3]/2)]
data = np2torch(data, "cuda:0").unsqueeze(0)
data = dataset.scale(data)

data_lr = laplace_pyramid_downscale2D(data, 2,
    opt['spatial_downscale_ratio'],
    "cuda:0", opt['periodic'])
data_lr = F.interpolate(data_lr, size=[64,64], mode='bilinear', align_corners=False)

data_lr_image = data_lr[0,:,:,:].cpu().numpy()
imageio.imwrite("lr.png", data_lr_image.swapaxes(0,1).swapaxes(1,2))

for i in range(data_lr_image.shape[0]):
    fourier_image = np.fft.fftn(data_lr_image[i])
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(data_lr_image.shape[1]) * data_lr_image.shape[1]
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, (data_lr_image.shape[1]/2) + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    import scipy.stats as stats
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)
    plt.plot(kvals, Abins, color="red", label="low-res channel " + str(i))

data_hr = laplace_pyramid_downscale2D(data, 1,
    opt['spatial_downscale_ratio'],
    "cuda:0", opt['periodic'])

data_hr_image = data_hr[0,:,:,:].cpu().numpy()
imageio.imwrite("hr.png", data_hr_image.swapaxes(0,1).swapaxes(1,2))

for i in range(data_hr_image.shape[0]):
    fourier_image = np.fft.fftn(data_hr_image[i])
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(data_hr_image.shape[1]) * data_hr_image.shape[1]
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, (data_hr_image.shape[1]/2) + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    import scipy.stats as stats
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)
    plt.plot(kvals, Abins, color="green", label="high-res channel " + str(i))
#x = generators[1].model[0](data_lr)
#x = generators[1].model[0][0](data_lr)
#x = generators[1].model[0][1](x)
#x = generators[1].model[1](x)
#x = generators[1].model[1][0](x)
#x = generators[1].model[1][1](x)
#x = generators[1].model[1](x)
#x = generators[1].model[2](x)
#x = generators[1].model[3](x)
#x = generators[1].model[4](x)
#x += data_lr
#x = generators[1](data_lr)
x = generators[1](data_lr, None)
#x = generate_by_patch(generators[0:2], 
#        "reconstruct", opt, 
#       opt['device'], opt['patch_size'], 
#        generated_image=data_lr, start_scale=1)
print(x.shape)
for i in range(x.shape[1]):
    chan_img = x[0,i,:,:].detach().cpu().numpy()
    imageio.imwrite("channel_"+str(i)+".png", chan_img)
    fourier_image = np.fft.fftn(chan_img)
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(data_lr_image.shape[1]) * data_lr_image.shape[1]
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, (data_lr_image.shape[1]/2) + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)
    plt.plot(kvals, Abins, color="blue", label="upscaled channel " + str(i))

plt.yscale("log")
plt.xlabel("k (wavenumber, pixels)")
plt.ylabel("P(K)")
plt.title("Power spectrum")
plt.legend()
plt.show()
 
imageio.imwrite("output.png", x[0].detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2))