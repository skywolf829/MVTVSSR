from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import time
import math
import random
import datetime
import os
from utility_functions import *
from options import *
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter
from matplotlib.pyplot import cm
from math import pi
from skimage.transform.pyramids import pyramid_reduce
from torch.utils.tensorboard import SummaryWriter
import copy
from pytorch_memlab import LineProfiler, MemReporter, profile, profile_every

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def TAD(field, device):
    if(field.shape[1] == 2):
        tx = spatial_derivative2D(field[:,0:1,:,:], 1, device)
        ty = spatial_derivative2D(field[:,1:2,:,:], 0, device)
        g = torch.abs(tx + ty)
    elif(field.shape[1] == 3):
        tx = spatial_derivative2D(field[:,0:1,:,:], 1, device)
        ty = spatial_derivative2D(field[:,1:2,:,:], 0, device)
        g = torch.abs(tx + ty)
    return g

def TAD3D(field, device):
    tx = spatial_derivative3D(field[:,0:1,:,:,:], 2, device)
    ty = spatial_derivative3D(field[:,1:2,:,:,:], 1, device)
    tz = spatial_derivative3D(field[:,2:3,:,:,:], 0, device)
    g = torch.abs(tx + ty + tz)
    return g

def curl2D(field, device):
    dydx = spatial_derivative2D(field[:,1:2], 0, device)
    dxdy = spatial_derivative2D(field[:,0:1], 1, device)
    output = dydx-dxdy
    return output

def curl3D(field, device):
    dzdy = spatial_derivative3D_CD(field[:,2:3], 1, device)
    dydz = spatial_derivative3D_CD(field[:,1:2], 2, device)
    dxdz = spatial_derivative3D_CD(field[:,0:1], 2, device)
    dzdx = spatial_derivative3D_CD(field[:,2:3], 0, device)
    dydx = spatial_derivative3D_CD(field[:,1:2], 0, device)
    dxdy = spatial_derivative3D_CD(field[:,0:1], 1, device)
    output = torch.cat((dzdy-dydz,dxdz-dzdx,dydx-dxdy), 1)
    return output

def curl3D8(field, device):
    dzdy = spatial_derivative3D_CD8(field[:,2:3], 1, device)
    dydz = spatial_derivative3D_CD8(field[:,1:2], 2, device)
    dxdz = spatial_derivative3D_CD8(field[:,0:1], 2, device)
    dzdx = spatial_derivative3D_CD8(field[:,2:3], 0, device)
    dydx = spatial_derivative3D_CD8(field[:,1:2], 0, device)
    dxdy = spatial_derivative3D_CD8(field[:,0:1], 1, device)
    output = torch.cat((dzdy-dydz,dxdz-dzdx,dydx-dxdy), 1)
    return output

def TAD3D_CD(field, device):
    tx = spatial_derivative3D_CD(field[:,0:1,:,:,:], 0, device)
    ty = spatial_derivative3D_CD(field[:,1:2,:,:,:], 1, device)
    tz = spatial_derivative3D_CD(field[:,2:3,:,:,:], 2, device)
    g = torch.abs(tx + ty + tz)
    return g

def TAD3D_CD8(field, device):
    tx = spatial_derivative3D_CD8(field[:,0:1,:,:,:], 0, device)
    ty = spatial_derivative3D_CD8(field[:,1:2,:,:,:], 1, device)
    tz = spatial_derivative3D_CD8(field[:,2:3,:,:,:], 2, device)
    g = torch.abs(tx + ty + tz)
    return g

def spatial_derivative2D_sobel(field, axis, device):
    m = nn.ReplicationPad2d(1)
    if(axis == 0):
        weights = torch.tensor(
            np.array([
            [-1/8, 0, 1/8], 
            [-1/4, 0, 1/4],
            [-1/8, 0, 1/8]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    elif(axis == 1):
        weights = torch.tensor(
            np.array([
            [-1/8, -1/4, -1/8], 
            [   0,    0,    0], 
            [ 1/8,  1/4,  1/8]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    return output

def spatial_derivative2D(field, axis, device):
    m = nn.ReplicationPad2d(1)
    if(axis == 0):
        weights = torch.tensor(
            np.array([
            [0, 0, 0], 
            [-0.5, 0, 0.5],
            [0, 0, 0]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    elif(axis == 1):
        weights = torch.tensor(
            np.array([
            [0, -0.5, 0], 
            [0,  0,   0], 
            [0, 0.5,  0]
            ]
        ).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    return output

def spatial_derivative3D_CD(field, axis, device):
    m = nn.ReplicationPad3d(1)
    # the first (a) axis in [a, b, c]
    if(axis == 0):
        weights = torch.tensor(np.array(
            [[[0, 0, 0], 
            [0, -0.5, 0],
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0.5, 0], 
            [0, 0, 0]]])
            .astype(np.float32)).to(device)
    elif(axis == 1):        
        # the second (b) axis in [a, b, c]
        weights = torch.tensor(np.array([
            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, -0.5, 0], 
            [0, 0, 0], 
            [0, 0.5, 0]],

            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]]])
            .astype(np.float32)).to(device)
    elif(axis == 2):
        # the third (c) axis in [a, b, c]
        weights = torch.tensor(np.array([
            [[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [-0.5, 0, 0.5], 
            [0, 0, 0]],

            [[0, 0, 0], 
            [0, 0,  0], 
            [ 0, 0, 0]]])
            .astype(np.float32)).to(device)
    weights = weights.view(1, 1, 3, 3, 3)
    field = m(field)
    output = F.conv3d(field, weights)
    return output

def spatial_derivative3D_CD8(field, axis, device):
    m = nn.ReplicationPad3d(4)
    # the first (a) axis in [a, b, c]
    if(axis == 0):
        weights = torch.zeros([9, 9, 9], dtype=torch.float32).to(device)
        weights[0, 4, 4] = 1/280
        weights[1, 4, 4] = -4/105
        weights[2, 4, 4] = 1/5
        weights[3, 4, 4] = -4/5
        weights[4, 4, 4] = 0
        weights[5, 4, 4] = 4/5
        weights[6, 4, 4] = -1/5
        weights[7, 4, 4] = 4/105
        weights[8, 4, 4] = -1/280
        
    elif(axis == 1):        
        # the second (b) axis in [a, b, c]
        weights = torch.zeros([9, 9, 9], dtype=torch.float32).to(device)
        weights[4, 0, 4] = 1/280
        weights[4, 1, 4] = -4/105
        weights[4, 2, 4] = 1/5
        weights[4, 3, 4] = -4/5
        weights[4, 4, 4] = 0
        weights[4, 5, 4] = 4/5
        weights[4, 6, 4] = -1/5
        weights[4, 7, 4] = 4/105
        weights[4, 8, 4] = -1/280
    elif(axis == 2):
        # the third (c) axis in [a, b, c]
        weights = torch.zeros([9, 9, 9], dtype=torch.float32).to(device)
        weights[4, 4, 1] = 1/280
        weights[4, 4, 1] = -4/105
        weights[4, 4, 2] = 1/5
        weights[4, 4, 3] = -4/5
        weights[4, 4, 4] = 0
        weights[4, 4, 5] = 4/5
        weights[4, 4, 6] = -1/5
        weights[4, 4, 7] = 4/105
        weights[4, 4, 8] = -1/280
    weights = weights.view(1, 1, 9, 9, 9)
    field = m(field)
    output = F.conv3d(field, weights)
    return output

def calc_gradient_penalty(discrim, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discrim(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def mag_difference(t1, t2):
    mag_1 = torch.zeros(t1.shape).to(t1.device)
    mag_2 = torch.zeros(t1.shape).to(t1.device)
    for i in range(t1.shape[1]):
        mag_1[0, 0] += t1[0, i]**2
        mag_2[0, 0] += t2[0, i]**2
    
    mag_1 = torch.sqrt(mag_1[0:, 0:1])
    mag_2 = torch.sqrt(mag_2[0:, 0:1])
    mag_diff = torch.abs(mag_2-mag_1)
    '''
    t_1 = t1*(1/torch.norm(t1, dim=1).view(1, 1, t1.shape[2], t1.shape[3]).repeat(1, t1.shape[1], 1, 1))
    t_2 = t2*(1/torch.norm(t2, dim=1).view(1, 1, t1.shape[2], t1.shape[3]).repeat(1, t1.shape[1], 1, 1))
    c = (t_1* t_2).sum(dim=1)

    angle_diff = torch.acos(c)
    angle_diff[angle_diff != angle_diff] = 0
    angle_diff = angle_diff.unsqueeze(0)    
    '''
    return mag_diff

def reflection_pad2D(frame, padding, device):
    frame = F.pad(frame, 
    [padding, padding, padding, padding])
    indices_to_fix = []
    for i in range(0, padding):
        indices_to_fix.append(i)
    for i in range(frame.shape[2] - padding, frame.shape[2]):
        indices_to_fix.append(i)

    for x in indices_to_fix:
        if(x < padding):
            correct_x = frame.shape[2] - 2*padding - x
        else:
            correct_x = x - frame.shape[2] + 2*padding
        for y in indices_to_fix:
            if(y < padding):
                correct_y = frame.shape[3] - 2*padding - y
            else:
                correct_y = y - frame.shape[3] + 2*padding
            frame[:, :, x, y] = frame[:, :, correct_x, correct_y]
    return frame

def reflection_pad3D(frame, padding, device):
    frame = F.pad(frame, 
    [padding, padding, padding, padding, padding, padding])
    indices_to_fix = []
    for i in range(0, padding):
        indices_to_fix.append(i)
    for i in range(frame.shape[2] - padding, frame.shape[2]):
        indices_to_fix.append(i)
    for x in indices_to_fix:
        if(x < padding):
            correct_x = frame.shape[2] - 2*padding - x
        else:
            correct_x = x - frame.shape[2] + 2*padding
        for y in indices_to_fix:
            if(y < padding):
                correct_y = frame.shape[3] - 2*padding - y
            else:
                correct_y = y - frame.shape[3] + 2*padding
            for z in indices_to_fix:
                if(z < padding):
                    correct_z = frame.shape[4] - 2*padding - z
                else:
                    correct_z = z - frame.shape[4] + 2*padding
                frame[:, :, x, y, z] = frame[:, :, correct_x, correct_y, correct_z]
    return frame

def laplace_pyramid_downscale2D(frame, level, downscale_per_level, device, periodic=False):
    kernel_size = 5
    sigma = 2 * (1 / downscale_per_level) / 6

    xy_grid = torch.zeros([kernel_size, kernel_size, 2])
    for i in range(kernel_size):
        for j in range(kernel_size):
                xy_grid[i, j, 0] = i
                xy_grid[i, j, 1] = j

    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)
    gaussian_kernel = gaussian_kernel.repeat(frame.shape[1], 1, 1, 1)
    input_size = np.array(list(frame.shape[2:]))
    with torch.no_grad():
        for i in range(level):
            s = (input_size * (downscale_per_level**(i+1))).astype(int)
            if(periodic):
                frame = reflection_pad2D(frame, int(kernel_size / 2), device)
            
            frame = F.conv2d(frame, gaussian_kernel, groups=frame.shape[1])
            frame = F.interpolate(frame, size = list(s), mode='bilinear', align_corners=False)
    del gaussian_kernel
    return frame

def laplace_pyramid_downscale3D(frame, level, downscale_per_level, device, periodic=False):
    kernel_size = 5
    sigma = 2 * (1 / downscale_per_level) / 6

    xyz_grid = torch.zeros([kernel_size, kernel_size, kernel_size, 3])
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                xyz_grid[i, j, k, 0] = i
                xyz_grid[i, j, k, 1] = j
                xyz_grid[i, j, k, 2] = k
   
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xyz_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, 
    kernel_size, kernel_size).to(device)
    gaussian_kernel = gaussian_kernel.repeat(frame.shape[1], 1, 1, 1, 1)
    input_size = np.array(list(frame.shape[2:]))
    
    with torch.no_grad():
        for i in range(level):
            s = (input_size * (downscale_per_level**(i+1))).astype(int)
            if(periodic):
                frame = reflection_pad3D(frame, int(kernel_size / 2), device)
            frame = F.conv3d(frame, gaussian_kernel, groups=frame.shape[1])
            frame = F.interpolate(frame, size = list(s), mode='trilinear', align_corners=False)
    del gaussian_kernel
    return frame
    
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def generate_padded_noise(size, pad_size, pad_with_noise, mode, device):
    if(pad_with_noise):
        for i in range(2,len(size)):
            size[i] += 2*pad_size
        noise = torch.randn(size, device=device)
    else:
        noise = torch.randn(size, device=device)
        if mode == "2D":
            required_padding = [pad_size, pad_size, pad_size, pad_size]
        else:
            required_padding = [pad_size, pad_size, pad_size, pad_size, pad_size, pad_size]
        noise = F.pad(noise, required_padding)
    return noise

def init_scales(opt, dataset):
    ns = []
    if(opt["spatial_downscale_ratio"] < 1.0):
        for i in range(len(dataset.resolution)):
            ns.append(round(math.log(opt["min_dimension_size"] / dataset.resolution[i]) / math.log(opt["spatial_downscale_ratio"]))+1)

    opt["n"] = min(ns)
    print("The model will have %i scales" % (opt["n"]))
    for i in range(opt["n"]):
        scaling = []
        factor =  opt["spatial_downscale_ratio"]**(opt["n"] - i - 1)
        for j in range(len(dataset.resolution)):
            x = int(dataset.resolution[j] * factor)
            scaling.append(x)
        #opt["resolutions"].insert(0, scaling)
        opt["resolutions"].append(scaling)
        print("Scale %i: %s" % (opt["n"] - 1 - i, str(scaling)))

def init_gen(scale, opt):
    num_kernels = int( 2** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 4)))

    generator = SinGAN_Generator(opt["resolutions"][scale], opt["num_blocks"], 
    opt["num_channels"], num_kernels, opt["kernel_size"], opt["stride"], 
    opt["pre_padding"], opt["mode"], opt["physical_constraints"], opt['separate_chans'], scale,
    opt['zero_noise'], opt["device"])

    generator.apply(weights_init)
    return generator, num_kernels

def init_discrim(scale, opt):
    num_kernels = int(2 ** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 4)))

    discriminator = SinGAN_Discriminator(opt["resolutions"][scale], 
    opt["num_blocks"], opt["num_channels"], num_kernels, opt["kernel_size"], 
    opt["stride"], opt['regularization'], opt["mode"],
    opt["device"])
    return discriminator

def generate(generators, mode, opt, device, generated_image=None, start_scale=0):
    with torch.no_grad():
        if(generated_image is None):
            generated_image = torch.zeros(generators[0].get_input_shape()).to(device)
        
        for i in range(0, len(generators)):
            generated_image = F.interpolate(generated_image, 
            size=generators[i].resolution, mode=opt["upsample_mode"], align_corners=False)
            
            if(mode == "reconstruct"):
                noise = generators[i].optimal_noise
            elif(mode == "random"):
                noise = torch.randn(generators[i].get_input_shape(), 
                device=device)
            generated_image = generators[i](generated_image, 
            opt["noise_amplitudes"][i+start_scale]*noise)
            

    return generated_image

def generate_by_patch(generators, mode, opt, device, patch_size, 
generated_image=None, start_scale=0):
    with torch.no_grad():
        #seq = []
        if(generated_image is None):
            generated_image = torch.zeros(generators[0].get_input_shape()).to(device)
        
        for i in range(start_scale, len(generators)):
            #print("Gen " + str(i))
            rf = int(generators[i].receptive_field() / 2)
            
            LR = F.interpolate(generated_image, 
            size=generators[i].resolution, mode=opt["upsample_mode"], align_corners=False)
            generated_image = torch.zeros(generators[i].get_input_shape()).to(device)

            if(mode == "reconstruct"):
                full_noise = generators[i].optimal_noise
            elif(mode == "random"):
                full_noise = torch.randn(generators[i].optimal_noise.shape, device=device)


            if(opt['mode'] == "2D" or opt['mode'] == "3Dto2D"):
                y_done = False
                y = 0
                y_stop = min(generated_image.shape[2], y + patch_size)
                while(not y_done):
                    if(y_stop == generated_image.shape[2]):
                        y_done = True
                    x_done = False
                    x = 0
                    x_stop = min(generated_image.shape[3], x + patch_size)
                    while(not x_done):                        
                        if(x_stop == generated_image.shape[3]):
                            x_done = True

                        noise = full_noise[:,:,y:y_stop,x:x_stop]

                        #print("[%i:%i, %i:%i, %i:%i]" % (z, z_stop, y, y_stop, x, x_stop))
                        result = generators[i](LR[:,:,y:y_stop,x:x_stop], 
                        opt["noise_amplitudes"][i]*noise)

                        x_offset = rf if x > 0 else 0
                        y_offset = rf if y > 0 else 0

                        generated_image[:,:,
                        y+y_offset:y+noise.shape[2],
                        x+x_offset:x+noise.shape[3]] = result[:,:,y_offset:,x_offset:]

                        x += patch_size - 2*rf
                        x = min(x, max(0, generated_image.shape[3] - patch_size))
                        x_stop = min(generated_image.shape[3], x + patch_size)
                    y += patch_size - 2*rf
                    y = min(y, max(0, generated_image.shape[2] - patch_size))
                    y_stop = min(generated_image.shape[2], y + patch_size)

        
            elif(opt['mode'] == '3D'):
                
                z_done = False
                z = 0
                z_stop = min(generated_image.shape[2], z + patch_size)
                while(not z_done):
                    if(z_stop == generated_image.shape[2]):
                        z_done = True
                    y_done = False
                    y = 0
                    y_stop = min(generated_image.shape[3], y + patch_size)
                    while(not y_done):
                        if(y_stop == generated_image.shape[3]):
                            y_done = True
                        x_done = False
                        x = 0
                        x_stop = min(generated_image.shape[4], x + patch_size)
                        while(not x_done):                        
                            if(x_stop == generated_image.shape[4]):
                                x_done = True

                            noise = full_noise[:,:,z:z_stop,y:y_stop,x:x_stop]

                            #print("[%i:%i, %i:%i, %i:%i]" % (z, z_stop, y, y_stop, x, x_stop))
                            result = generators[i](LR[:,:,z:z_stop,y:y_stop,x:x_stop], 
                            opt["noise_amplitudes"][i]*noise)

                            x_offset = rf if x > 0 else 0
                            y_offset = rf if y > 0 else 0
                            z_offset = rf if z > 0 else 0

                            generated_image[:,:,
                            z+z_offset:z+noise.shape[2],
                            y+y_offset:y+noise.shape[3],
                            x+x_offset:x+noise.shape[4]] = result[:,:,z_offset:,y_offset:,x_offset:]

                            x += patch_size - 2*rf
                            x = min(x, max(0, generated_image.shape[4] - patch_size))
                            x_stop = min(generated_image.shape[4], x + patch_size)
                        y += patch_size - 2*rf
                        y = min(y, max(0, generated_image.shape[3] - patch_size))
                        y_stop = min(generated_image.shape[3], y + patch_size)
                    z += patch_size - 2*rf
                    z = min(z, max(0, generated_image.shape[2] - patch_size))
                    z_stop = min(generated_image.shape[2], z + patch_size)


    #seq.append(generated_image.detach().cpu().numpy()[0].swapaxes(0,2).swapaxes(0,1))
    #seq = np.array(seq)
    #seq -= seq.min()
    #seq /= seq.max()
    #seq *= 255
    #seq = seq.astype(np.uint8)
    #imageio.mimwrite("patches_good.gif", seq)
    #imageio.imwrite("patch_good_ex0.png", seq[0,0:100, 0:100,:])
    #imageio.imwrite("patch_good_ex1.png", seq[1,0:100, 0:100,:])
    #imageio.imwrite("patch_good_ex2.png", seq[2,0:100, 0:100,:])
    return generated_image

def super_resolution(generator, frame, factor, opt, device):
    
    frame = frame.to(device)
    full_size = list(frame.shape[2:])
    for i in range(len(full_size)):
        full_size[i] *= factor
    r = 1 / opt["spatial_downscale_ratio"]
    curr_r = 1.0
    while(curr_r * r < factor):
        frame = F.interpolate(frame, scale_factor=r,mode=opt["upsample_mode"], align_corners=False)
        noise = torch.randn(frame.shape).to(device)
        frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
        curr_r *= r
    frame = F.interpolate(frame, size=full_size, mode=opt["upsample_mode"], align_corners=False)
    noise = torch.randn(frame.shape).to(device)
    noise = torch.zeros(frame.shape).to(device)
    frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
    return frame

def save_models(generators, discriminators, opt, optimizer=None):
    folder = create_folder(opt["save_folder"], opt["save_name"])
    path_to_save = os.path.join(opt["save_folder"], folder)
    print_to_log_and_console("Saving model to %s" % (path_to_save), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt["save_generators"]):
        optimal_noises = {}
        gen_states = {}
        
        for i in range(len(generators)):
            gen_states[str(i)] = generators[i].state_dict()
            optimal_noises[str(i)] = generators[i].optimal_noise
        torch.save(gen_states, os.path.join(path_to_save, "SinGAN.generators"))
        torch.save(optimal_noises, os.path.join(path_to_save, "SinGAN.optimal_noises"))

    if(opt["save_discriminators"]):
        discrim_states = {}
        for i in range(len(discriminators)):
            discrim_states[str(i)] = discriminators[i].state_dict()
        torch.save(discrim_states, os.path.join(path_to_save, "SinGAN.discriminators"))

    save_options(opt, path_to_save)

def load_models(opt, device):
    generators = []
    discriminators = []
    load_folder = os.path.join(opt["save_folder"], opt["save_name"])

    if not os.path.exists(load_folder):
        print_to_log_and_console("%s doesn't exist, load failed" % load_folder, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(load_folder, "SinGAN.generators")):
        gen_params = torch.load(os.path.join(load_folder, "SinGAN.generators"),
        map_location=device)
        optimal_noises = torch.load(os.path.join(load_folder, "SinGAN.optimal_noises"),
        map_location=device)
        for i in range(opt["n"]):
            if(str(i) in gen_params.keys()):
                gen_params_compat = OrderedDict()
                for k, v in gen_params[str(i)].items():
                    if("module" in k):
                        gen_params_compat[k[7:]] = v
                    else:
                        gen_params_compat[k] = v
                generator, num_kernels = init_gen(i, opt)
                generator.optimal_noise = optimal_noises[str(i)]
                generator.load_state_dict(gen_params_compat)
                generators.append(generator)

        print_to_log_and_console("Successfully loaded SinGAN.generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "MVTVSSRGAN.generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if os.path.exists(os.path.join(load_folder, "SinGAN.discriminators")):
        discrim_params = torch.load(os.path.join(load_folder, "SinGAN.discriminators"),
        map_location=device)
        for i in range(opt["n"]):
            if(str(i) in discrim_params.keys()):
                discrim_params_compat = OrderedDict()
                for k, v in discrim_params[str(i)].items():
                    if(k[0:7] == "module."):
                        discrim_params_compat[k[7:]] = v
                    else:
                        discrim_params_compat[k] = v
                discriminator = init_discrim(i, opt)
                discriminator.load_state_dict(discrim_params_compat)
                discriminators.append(discriminator)
        print_to_log_and_console("Successfully loaded SinGAN.discriminators", 
        os.path.join(opt["save_folder"],opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "MVTVSSRGAN.s_discriminators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    
    return  generators, discriminators

def train_single_scale_wrapper(generators, discriminators, opt):
    with LineProfiler(train_single_scale, generate, generate_by_patch, SinGAN_Generator.forward) as prof:
        g, d = train_single_scale(generators, discriminators, opt)
    print(prof.display())
    return g, d

def train_single_scale(generators, discriminators, opt):
    
    start_t = time.time()
    # Initialize the dataset
    dataset = Dataset(os.path.join(input_folder, opt["data_folder"]), opt)

    torch.manual_seed(0)
    
    # Create the new generator and discriminator for this level
    if(len(generators) == opt['scale_in_training']):
        generator, num_kernels_this_scale = init_gen(len(generators), opt)
        generator = generator.to(opt["device"])
        discriminator = init_discrim(len(generators), opt).to(opt["device"])
    else:
        generator = generators[-1].to(opt['device'])
        generators.pop(len(generators)-1)
        discriminator = discriminators[-1].to(opt['device'])
        discriminators.pop(len(discriminators)-1)

    # Move all models to this GPU and make them distributed
    for i in range(len(generators)):
        generators[i].to(opt["device"])
        generators[i].eval()
        for param in generators[i].parameters():
            param.requires_grad = False

    #print_to_log_and_console(generator, os.path.join(opt["save_folder"], opt["save_name"]),
    #    "log.txt")
    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    #print_to_log_and_console("Kernels this scale: %i" % num_kernels_this_scale, 
    #    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    generator_optimizer = optim.Adam(generator.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,
    milestones=[1600-opt['iteration_number']],gamma=opt['gamma'])

    discriminator_optimizer = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), 
    lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_optimizer,
    milestones=[1600-opt['iteration_number']],gamma=opt['gamma'])
 
    writer = SummaryWriter(os.path.join('tensorboard',opt['save_name']))

    start_time = time.time()
    next_save = 0
    images_seen = 0

    # Get properly sized frame for this generator
    real = dataset.__getitem__(0)
    real = real.to(opt["device"])
    print(str(len(generators)) + ": " + str(opt["resolutions"][len(generators)]))
    
    if(len(generators) + 1 is not len(opt["resolutions"])):
        if(opt['mode'] == '2D' or opt['mode'] == '3Dto2D'):
            real = laplace_pyramid_downscale2D(real, len(opt['resolutions'])-len(generators)-1, 
            opt['spatial_downscale_ratio'], opt['device'], opt['periodic'])
        elif(opt['mode'] == "3D"):
            real = laplace_pyramid_downscale3D(real, len(opt['resolutions'])-len(generators)-1, 
            opt['spatial_downscale_ratio'], opt['device'], opt['periodic'])
    else:
        real = real.to(opt['device'])

    if(opt['mode'] == '3D'):
        max_dim = opt["training_patch_size"]*opt["training_patch_size"]*opt["training_patch_size"]
    elif(opt['mode'] == '2D' or opt['mode'] == '3Dto2D'):
        max_dim = opt["training_patch_size"]*opt["training_patch_size"]

    curr_size = opt["resolutions"][len(generators)][0]
    for z in range(1, len(opt["resolutions"][len(generators)])):
        curr_size *= opt["resolutions"][len(generators)][z]

    if(len(opt['noise_amplitudes']) <= len(generators)):
        opt["noise_amplitudes"].append(1.0)
        if(len(generators) > 0):
            optimal_LR = generate_by_patch(generators, "reconstruct", opt, 
            opt["device"], opt["patch_size"])
            optimal_LR = F.interpolate(optimal_LR, size=opt["resolutions"][len(generators)],
            mode=opt["upsample_mode"], align_corners=False)
            criterion = nn.MSELoss().to(opt['device'])
            rmse = torch.sqrt(criterion(optimal_LR, real))
            opt["noise_amplitudes"][-1] = rmse.item()
        else:    
            optimal_LR = torch.zeros(generator.get_input_shape(), device=opt["device"])
    else:
        if(len(generators) > 0):
            optimal_LR = generate_by_patch(generators, "reconstruct", opt, 
            opt["device"], opt["patch_size"])
            optimal_LR = F.interpolate(optimal_LR, size=opt["resolutions"][len(generators)],
            mode=opt["upsample_mode"], align_corners=False)
        else:    
            optimal_LR = torch.zeros(generator.get_input_shape(), device=opt["device"])

    if(curr_size > max_dim):
        starts_all, ends_all = dataset.get_patch_ranges(real, opt["training_patch_size"], 
        int(generator.receptive_field() / 2), opt['mode'])
    else:
        starts_all = [list(np.array(real.shape[2:]) * 0)]
        ends_all = [list(np.array(real.shape[2:]))]

    for epoch in range(opt['iteration_number'], opt["epochs"]):

        for patch_in_training in range(len(starts_all)):
            D_loss = 0
            G_loss = 0        
            gradient_loss = 0
            rec_loss = 0        
            g = 0
            mags = np.zeros(1)
            angles = np.zeros(1)

            starts = []
            ends = []
            
            starts = starts_all[patch_in_training]
            ends = ends_all[patch_in_training]
            if(opt['mode'] == "2D" or opt['mode'] == '3Dto2D'):
                r = real[:,:,starts[0]:ends[0],starts[1]:ends[1]]
            elif(opt['mode'] == "3D"):
                r = real[:,:,starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]
            
            
            # Update discriminator: maximize D(x) + D(G(z))
            if(opt["alpha_2"] > 0.0):            
                for j in range(opt["discriminator_steps"]):
                    discriminator.zero_grad()
                    generator.zero_grad()
                    D_loss = 0

                    # Train with real downscaled to this scale
                    output_real = discriminator(r)
                    discrim_error_real = -output_real.mean()
                    D_loss += discrim_error_real.mean().item()
                    discrim_error_real.backward(retain_graph=True)

                    # Train with the generated image
                    if(len(generators) > 0):
                        fake_prev = generate_by_patch(generators, "random", opt, 
                        opt["device"], opt["patch_size"])
                        fake_prev = F.interpolate(fake_prev, size=opt["resolutions"][len(generators)],
                        mode=opt["upsample_mode"], align_corners=False)
                    else:
                        fake_prev = torch.zeros(generator.get_input_shape()).to(opt["device"])

                    if(opt['mode'] == "2D" or opt['mode'] == '3Dto2D'):
                        fake_prev_view = fake_prev[:,:,starts[0]:ends[0],starts[1]:ends[1]]
                        noise = opt["noise_amplitudes"][-1] * torch.randn(fake_prev_view.shape,device=opt['device'])
                    elif(opt['mode'] == "3D"):
                        fake_prev_view = fake_prev[:,:,starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]
                        noise = opt["noise_amplitudes"][-1] * torch.randn(fake_prev_view.shape, device=opt['device'])
                    fake = generator(fake_prev_view, noise.detach())
                    output_fake = discriminator(fake.detach())
                    discrim_error_fake = output_fake.mean()
                    discrim_error_fake.backward(retain_graph=True)
                    D_loss += discrim_error_fake.item()

                    if(opt['regularization'] == "GP"):
                        # Gradient penalty 
                        gradient_penalty = calc_gradient_penalty(discriminator, r, fake, 1, opt['device'])
                        gradient_penalty.backward()
                    elif(opt['regularization'] == "TV"):
                        # Total variance penalty
                        TV_penalty_real = torch.abs(-discrim_error_real-1)
                        TV_penalty_real.backward(retain_graph=True)
                        TV_penalty_fake = torch.abs(discrim_error_fake+1)
                        TV_penalty_fake.backward(retain_graph=True)
                    discriminator_optimizer.step()

            # Update generator: maximize D(G(z))
            for j in range(opt["generator_steps"]):
                generator.zero_grad()
                discriminator.zero_grad()
                G_loss = 0
                gen_err_total = 0
                phys_loss = 0
                path_loss = 0
                loss = nn.L1Loss().to(opt["device"])
                if(opt["alpha_2"] > 0.0):
                    fake = generator(fake_prev_view, noise.detach())
                    output = discriminator(fake)
                    generator_error = -output.mean()# * opt["alpha_2"]
                    generator_error.backward(retain_graph=True)
                    gen_err_total += generator_error.mean().item()
                    G_loss = output.mean().item()
                if(opt['alpha_1'] > 0.0 or opt['alpha_4'] > 0.0):
                    if(opt['mode'] == "2D" or opt['mode'] == '3Dto2D'):
                        opt_noise = opt["noise_amplitudes"][-1]*generator.optimal_noise[:,:,starts[0]:ends[0],starts[1]:ends[1]]
                        optimal_reconstruction = generator(optimal_LR.detach()[:,:,starts[0]:ends[0],starts[1]:ends[1]], 
                        opt_noise)
                    elif(opt['mode'] == "3D"):
                        opt_noise = opt["noise_amplitudes"][-1]*generator.optimal_noise[:,:,starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]]
                        optimal_reconstruction = generator(optimal_LR.detach()[:,:,starts[0]:ends[0],starts[1]:ends[1],starts[2]:ends[2]], 
                        opt_noise)
                if(opt['alpha_1'] > 0.0):
                    rec_loss = loss(optimal_reconstruction, r) * opt["alpha_1"]
                    rec_loss.backward(retain_graph=True)
                    gen_err_total += rec_loss.item()
                    rec_loss = rec_loss.detach()
                if(opt['alpha_3'] > 0.0):
                    if(opt["physical_constraints"] == "soft"):
                        if(opt['mode'] == "2D" or opt['mode'] == '3Dto2D'):
                            g_map = TAD(dataset.unscale(optimal_reconstruction), opt["device"])            
                            g = g_map.mean()
                        elif(opt['mode'] == "3D"):
                            g_map = TAD3D_CD(dataset.unscale(optimal_reconstruction), opt["device"])
                            g = g_map.mean()
                        phys_loss = opt["alpha_3"] * g 
                        phys_loss.backward(retain_graph=True)
                        gen_err_total += phys_loss.item()
                        phys_loss = phys_loss.item()
                if(opt['alpha_4'] > 0.0):
                    
                    cs = torch.nn.CosineSimilarity(dim=1).to(opt['device'])
                    mags = torch.abs(torch.norm(optimal_reconstruction, dim=1) - torch.norm(r, dim=1))
                    angles = torch.abs(cs(optimal_reconstruction, r) - 1) / 2
                    r_loss = opt['alpha_4'] * (mags.mean() + angles.mean()) / 2
                    r_loss.backward(retain_graph=True)
                    gen_err_total += r_loss.item()
                if(opt['alpha_5'] > 0.0):
                    real_gradient = []
                    rec_gradient = []
                    for ax1 in range(r.shape[1]):
                        for ax2 in range(len(r.shape[2:])):
                            if(opt["mode"] == '2D' or opt['mode'] == '3Dto2D'):
                                r_deriv = spatial_derivative2D(r[:,ax1:ax1+1], ax2, opt['device'])
                                rec_deriv = spatial_derivative2D(optimal_reconstruction[:,ax1:ax1+1], ax2, opt['device'])
                            elif(opt['mode'] == '3D'):
                                r_deriv = spatial_derivative3D_CD(r[:,ax1:ax1+1], ax2, opt['device'])
                                rec_deriv = spatial_derivative3D_CD(optimal_reconstruction[:,ax1:ax1+1], ax2, opt['device'])
                            real_gradient.append(r_deriv)
                            rec_gradient.append(rec_deriv)
                    real_gradient = torch.cat(real_gradient, 1)
                    rec_gradient = torch.cat(rec_gradient, 1)
                    gradient_loss = loss(real_gradient, rec_gradient)
                    gradient_loss_adj = gradient_loss * opt['alpha_5']
                    gradient_loss_adj.backward(retain_graph=True)
                    gen_err_total += gradient_loss_adj.item()
                if(opt["alpha_6"] > 0):
                    if(opt['mode'] == '3D'):
                        if(opt['adaptive_streamlines']):
                            path_loss = adaptive_streamline_loss3D(r, optimal_reconstruction, 
                            torch.abs(mags[0] + angles[0]), int(opt['streamline_res']**3), 
                            3, 1, opt['streamline_length'], opt['device'], 
                            periodic=opt['periodic'])* opt['alpha_6']
                        else:
                            path_loss = streamline_loss3D(r, optimal_reconstruction, 
                            opt['streamline_res'], opt['streamline_res'], opt['streamline_res'], 
                            1, opt['streamline_length'], opt['device'], 
                            periodic=opt['periodic'] and optimal_reconstruction.shape == real.shape) * opt['alpha_6']
                    elif(opt['mode'] == '2D' or opt['mode'] == '3Dto2D'):
                        path_loss = streamline_loss2D(r, optimal_reconstruction, 
                        opt['streamline_res'], opt['streamline_res'], 
                        1, opt['streamline_length'], opt['device'], periodic=opt['periodic']) * opt['alpha_6']
                    path_loss.backward(retain_graph=True)
                    path_loss = path_loss.item()
                
                generator_optimizer.step()
        
        if(epoch % 50 == 0):
            if(opt['alpha_1'] > 0.0 or opt['alpha_4'] > 0.0):
                rec_numpy = optimal_reconstruction.detach().cpu().numpy()[0]
                rec_cm = toImg(rec_numpy)
                rec_cm -= rec_cm.min()
                rec_cm *= (1/rec_cm.max())
                writer.add_image("reconstructed/%i"%len(generators), 
                rec_cm.clip(0,1), epoch)

            real_numpy = r.detach().cpu().numpy()[0]
            real_cm = toImg(real_numpy)
            real_cm -= real_cm.min()
            real_cm *= (1/real_cm.max())
            writer.add_image("real/%i"%len(generators), 
            real_cm.clip(0,1), epoch)

            if(opt["alpha_2"] > 0.0):                
                fake_numpy = fake.detach().cpu().numpy()[0]
                fake_cm = toImg(fake_numpy)
                fake_cm -= fake_cm.min()
                fake_cm *= (1/fake_cm.max())
                writer.add_image("fake/%i"%len(generators), 
                fake_cm.clip(0,1), epoch)

                discrim_output_map_real_img = toImg(output_real.detach().cpu().numpy()[0])     
                writer.add_image("discrim_map_real/%i"%len(generators), 
                discrim_output_map_real_img, epoch)

                discrim_output_map_fake_img = toImg(output.detach().cpu().numpy()[0])
                writer.add_image("discrim_map_fake/%i"%len(generators), 
                discrim_output_map_fake_img, epoch)
            if(opt["alpha_4"] > 0.0):
                angles_cm = toImg(angles.detach().cpu().numpy())
                writer.add_image("angle/%i"%len(generators), 
                angles_cm , epoch)

                mags_cm = toImg(mags.detach().cpu().numpy())
                writer.add_image("mag/%i"%len(generators), 
                mags_cm, epoch)
            if(opt["alpha_3"] > 0.0):
                g_cm = toImg(g_map.detach().cpu().numpy()[0])
                writer.add_image("Divergence/%i"%len(generators), 
                g_cm, epoch)

        if(epoch % opt['save_every'] == 0):
            opt["iteration_number"] = epoch
            save_models(generators + [generator], discriminators + [discriminator], opt)
            

        print_to_log_and_console("%i/%i: Dloss=%.02f Gloss=%.02f L1=%.04f AMD=%.02f AAD=%.02f" %
        (epoch, opt['epochs'], D_loss, G_loss, rec_loss, mags.mean(), angles.mean()), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

        writer.add_scalar('D_loss_scale/%i'%len(generators), D_loss, epoch) 
        writer.add_scalar('G_loss_scale/%i'%len(generators), G_loss, epoch) 
        writer.add_scalar('L1/%i'%len(generators), rec_loss, epoch)
        writer.add_scalar('Gradient_loss/%i'%len(generators), gradient_loss / (opt['alpha_5']+1e-6), epoch)
        writer.add_scalar('TAD/%i'%len(generators), phys_loss / (opt["alpha_3"]+1e-6), epoch)
        writer.add_scalar('path_loss/%i'%len(generators), path_loss / (opt['alpha_6']+1e-6), epoch)
        writer.add_scalar('Mag_loss_scale/%i'%len(generators), mags.mean(), epoch) 
        writer.add_scalar('Angle_loss_scale/%i'%len(generators), angles.mean(), epoch) 

        discriminator_scheduler.step()
        generator_scheduler.step()

    generator = reset_grads(generator, False)
    generator.eval()
    discriminator = reset_grads(discriminator, False)
    discriminator.eval()

    return generator, discriminator

class SinGAN_Generator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, kernel_size,
    stride, pre_padding, mode, physical_constraints, separate_chans, scale, zero_noise, device):
        super(SinGAN_Generator, self).__init__()
        self.scale = scale
        self.pre_padding = pre_padding
        self.resolution = resolution
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.mode = mode
        self.zero_noise = zero_noise
        if(self.scale == 0 or not zero_noise):
            self.optimal_noise = torch.randn(self.get_input_shape(), device=device)
        else:
            self.optimal_noise = torch.zeros(self.get_input_shape(), device=device)
        self.physical_constraints = physical_constraints
        self.device = device
        self.separate_chans = separate_chans

        if(physical_constraints == "hard" and (mode == "2D" or mode =="3Dto2D")):
            output_chans = 1
        else:
            output_chans = num_channels

        if(pre_padding):
            pad_amount = int(kernel_size/2)
            self.layer_padding = 0
        else:
            pad_amount = 0
            self.layer_padding = 1

        if(mode == "2D" or mode == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            self.required_padding = [pad_amount, pad_amount, pad_amount, pad_amount]
            self.upscale_method = "bicubic"
        elif(mode == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d
            self.required_padding = [pad_amount, pad_amount, pad_amount, 
            pad_amount, pad_amount, pad_amount]
            self.upscale_method = "trilinear"

        if(not separate_chans):
            self.model = self.create_model(num_blocks, num_channels, output_chans,
            num_kernels, kernel_size, stride, 1,
            conv_layer, batchnorm_layer).to(device)
        else:
            self.model = self.create_model(num_blocks, num_channels, output_chans, 
            num_kernels, kernel_size, stride, num_channels,
            conv_layer, batchnorm_layer).to(device)

    def create_model(self, num_blocks, num_channels, output_chans,
    num_kernels, kernel_size, stride, groups, conv_layer, batchnorm_layer):
        modules = []
        
        for i in range(num_blocks):
            # The head goes from numChannels channels to numKernels
            if i == 0:
                modules.append(nn.Sequential(
                    conv_layer(num_channels, num_kernels*groups, kernel_size=kernel_size, 
                    stride=stride, padding=self.layer_padding, groups=groups),
                    batchnorm_layer(num_kernels*groups),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from kernel_size to num_channels before tanh [-1,1]
            elif i == num_blocks-1:  
                tail = nn.Sequential(
                    conv_layer(num_kernels*groups, output_chans, kernel_size=kernel_size, 
                    stride=stride, padding=self.layer_padding, groups=groups),
                    nn.Tanh()
                )              
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    conv_layer(num_kernels*groups, num_kernels*groups, kernel_size=kernel_size,
                    stride=stride, padding=self.layer_padding, groups=groups),
                    batchnorm_layer(num_kernels*groups),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        m = nn.Sequential(*modules)
        return m
        
    def get_input_shape(self):
        shape = []
        shape.append(1)
        shape.append(self.num_channels)
        for i in range(len(self.resolution)):
            shape.append(self.resolution[i])
        return shape

    def get_params(self):
        if(self.separate_chans):
            p = []
            for i in range(self.num_channels):
                p = p + list(self.model[i].parameters())
            return p
        else:
            return self.model.parameters()

    def receptive_field(self):
        return (self.kernel_size-1)*self.num_blocks

    def forward(self, data, noise=None):
        if(noise is None):
            noise = torch.zeros(data.shape).to(self.device)
        noisePlusData = data + noise
        if(self.pre_padding):
            noisePlusData = F.pad(noisePlusData, self.required_padding)
        output = self.model(noisePlusData)

        if(self.physical_constraints == "hard" and self.mode == '3D'):
            output = curl3D(output, self.device)
            return output
        elif(self.physical_constraints == "hard" and (self.mode == '2D' or self.mode == '3Dto2D')):
            output = curl2D(output, self.device)
            gradx = spatial_derivative2D(output[:,0:1], 0, self.device)
            grady = spatial_derivative2D(output[:,1:2], 1, self.device)
            output = torch.cat([-grady, gradx], axis=1)
            return output
        else:
            return output + data

class SinGAN_Discriminator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, 
    kernel_size, stride, regularization, mode, device):
        super(SinGAN_Discriminator, self).__init__()
        self.device=device
        modules = []
        self.resolution = resolution
        self.mode = mode
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        use_sn = regularization == "SN"
        if(mode == "2D" or mode == "3Dto2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(mode == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d

        for i in range(num_blocks):
            # The head goes from 3 channels (RGB) to num_kernels
            if i == 0:
                modules.append(nn.Sequential(
                    create_conv_layer(conv_layer, num_channels, num_kernels, 
                    kernel_size, stride, 0, use_sn),
                    create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from num_kernels to 1 channel for discriminator optimization
            elif i == num_blocks-1:  
                tail = nn.Sequential(
                    create_conv_layer(conv_layer, num_kernels, 1, 
                    kernel_size, stride, 0, use_sn)
                )
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    create_conv_layer(conv_layer, num_kernels, num_kernels, 
                    kernel_size, stride, 0, use_sn),
                    create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model =  nn.Sequential(*modules)
        self.model = self.model.to(device)

    def receptive_field(self):
        return (self.kernel_size-1)*self.num_blocks

    def forward(self, x):
        return self.model(x)

def create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn):
    bnl = batchnorm_layer(num_kernels)
    bnl.apply(weights_init)
    if(use_sn):
        bnl = SpectralNorm(bnl)
    return bnl

def create_conv_layer(conv_layer, in_chan, out_chan, kernel_size, stride, padding, use_sn):
    c = conv_layer(in_chan, out_chan, 
                    kernel_size, stride, 0)
    c.apply(weights_init)
    if(use_sn):
        c = SpectralNorm(c)
    return c

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location, opt):
        self.dataset_location = dataset_location
        self.channel_mins = []
        self.channel_maxs = []
        self.num_items = 0
        self.image_normalize = opt["image_normalize"]
        self.scale_data = opt["scale_data"]
        self.scale_on_magnitude = opt['scale_on_magnitude']
        self.max_mag = None
        self.mode = opt['mode']
        for filename in os.listdir(self.dataset_location):
            self.num_items += 1
            d = np.load(os.path.join(self.dataset_location, filename))
            self.num_channels = d.shape[0]
            self.resolution = d.shape[1:]
            if(self.mode == "3Dto2D"):
                self.resolution = self.resolution[0:len(self.resolution)-1]
            for i in range(d.shape[0]):
                mags = np.linalg.norm(d, axis=0)
                m_mag = mags.max()
                if(self.max_mag is None or self.max_mag < m_mag):
                    self.max_mag = m_mag
                if(len(self.channel_mins) <= i):
                    self.channel_mins.append(d[i].min())
                    self.channel_maxs.append(d[i].max())
                else:
                    if(d[i].max() > self.channel_maxs[i]):
                        self.channel_maxs[i] = d[i].max()
                    if(d[i].min() < self.channel_mins[i]):
                        self.channel_mins[i] = d[i].min()
        #print(self.channel_mins)
        #print(self.channel_maxs)
    def __len__(self):
        return self.num_items

    def resolution(self):
        return self.resolution

    def scale(self, data):
        d = data.clone()
        if(self.scale_on_magnitude):
            d *= (1/self.max_mag)
        return d
    def unscale(self, data):
        d = data.clone()
        if(self.scale_data):
            for i in range(self.num_channels):
                d[0, i] *= 0.5
                d[0, i] += 0.5
                d[0, i] *= (self.channel_maxs[i] - self.channel_mins[i])
                d[0, i] += self.channel_mins[i]
        elif(self.scale_on_magnitude):
            d *= self.max_mag
        return d

    def get_patch_ranges(self, frame, patch_size, receptive_field, mode):
        starts = []
        rf = receptive_field
        ends = []
        if(mode == "3D"):
            for z in range(0,max(1,frame.shape[2]), patch_size-2*rf):
                z = min(z, max(0, frame.shape[2] - patch_size))
                z_stop = min(frame.shape[2], z + patch_size)
                
                for y in range(0, max(1,frame.shape[3]), patch_size-2*rf):
                    y = min(y, max(0, frame.shape[3] - patch_size))
                    y_stop = min(frame.shape[3], y + patch_size)

                    for x in range(0, max(1,frame.shape[4]), patch_size-2*rf):
                        x = min(x, max(0, frame.shape[4] - patch_size))
                        x_stop = min(frame.shape[4], x + patch_size)

                        starts.append([z, y, x])
                        ends.append([z_stop, y_stop, x_stop])
        elif(mode == "2D" or mode == "3Dto2D"):
            for y in range(0, max(1,frame.shape[2]-patch_size+1), patch_size-2*rf):
                y = min(y, max(0, frame.shape[2] - patch_size))
                y_stop = min(frame.shape[2], y + patch_size)

                for x in range(0, max(1,frame.shape[3]-patch_size+1), patch_size-2*rf):
                    x = min(x, max(0, frame.shape[3] - patch_size))
                    x_stop = min(frame.shape[3], x + patch_size)

                    starts.append([y, x])
                    ends.append([y_stop, x_stop])
        return starts, ends

    def __getitem__(self, index):
        data = np.load(os.path.join(self.dataset_location, str(index) + ".npy"))
        if(self.image_normalize):
            data = data.astype(np.float32) / 255
            data -= 0.5
            data *= 2
        elif(self.scale_data):
            for i in range(self.num_channels):
                data[i] -= self.channel_mins[i]
                data[i] *= (1 / (self.channel_maxs[i] - self.channel_mins[i]))
                data[i] -= 0.5
                data[i] *= 2
        elif(self.scale_on_magnitude):
            data *= (1 / self.max_mag)
            
        if(self.mode == "3Dto2D"):
            data = data[:,:,:,int(data.shape[3]/2)]
        data = np2torch(data, "cpu")
        return data.unsqueeze(0)