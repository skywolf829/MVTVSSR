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
from torch.utils.tensorboard import SummaryWriter

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
    tx = spatial_derivative2D(field[:,0:1,:,:], "x", device)
    ty = spatial_derivative2D(field[:,1:2,:,:], "y", device)
    g = torch.abs(tx + ty)
    return g.sum()

def spatial_derivative2D(field, axis, device):
    m = nn.ReplicationPad2d(1)
    if(axis == "x"):
        weights = torch.tensor(np.array([[0, 0, 0], [-1/2, 0, 1/2], [0, 0, 0]]).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    elif(axis == "y"):
        weights = torch.tensor(np.array([[0, -1/2, 0], [0, 0, 0], [0, 1/2, 0]]).astype(np.float32)).to(device)
        weights = weights.view(1, 1, 3, 3)
        field = m(field)
        output = F.conv2d(field, weights)
    return output

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

def init_scales(opt):
    ns = []
    if(opt["spatial_downscale_ratio"] < 1.0):
        for i in range(len(opt["base_resolution"])):            
            ns.append(round(math.log(opt["min_dimension_size"] / opt["base_resolution"][i]) / math.log(opt["spatial_downscale_ratio"]))+1)

    opt["n"] = min(ns)
    print("The model will have %i scales" % (opt["n"]))
    for i in range(opt["n"]):
        scaling = []
        for j in range(len(opt["base_resolution"])):
            x = round(opt["base_resolution"][j] * (opt["spatial_downscale_ratio"] ** i))
            scaling.append(x)
        opt["resolutions"].insert(0, scaling)
        print("Scale %i: %s" % (opt["n"] - 1 - i, str(scaling)))

def init_gen(scale, opt):
    num_kernels = int( 2** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 4)))

    generator = SinGAN_Generator(opt["resolutions"][scale], opt["num_blocks"], 
    opt["num_channels"], num_kernels, opt["kernel_size"], opt["stride"], 
    opt["pre_padding"], opt["mode"], opt["physical_constraints"], opt["device"])

    weights_init(generator)
    return generator, num_kernels

def init_discrim(scale, opt):
    num_kernels = int(2 ** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 4)))

    discriminator = SinGAN_Discriminator(opt["resolutions"][scale], 
    opt["num_blocks"], opt["num_channels"], num_kernels, opt["kernel_size"], 
    opt["stride"], opt["use_spectral_norm"], opt["mode"],
    opt["device"])
    weights_init(discriminator)
    return discriminator

def generate(generators, mode, opt, device):
    generated_image = torch.zeros(generators[0].get_input_shape()).to(device)
    
    for i in range(0, len(generators)):
        generated_image = F.interpolate(generated_image, 
        size=generators[i].resolution, mode=opt["upsample_mode"])
        
        if(mode == "reconstruct"):
            noise = generators[i].optimal_noise
        elif(mode == "random"):
            noise = torch.randn(generators[i].get_input_shape(), 
            device=device)

        generated_image = generators[i](generated_image, 
        opt["noise_amplitudes"][i]*noise)

    return generated_image

def super_resolution(generator, frame, factor, opt, device):
    
    frame = frame.to(device)
    full_size = list(frame.shape[2:])
    for i in range(len(full_size)):
        full_size[i] *= factor
    r = 1 / opt["spatial_downscale_ratio"]
    curr_r = 1.0
    while(curr_r * r < factor):
        print(frame.shape)
        frame = F.interpolate(frame, scale_factor=r,mode=opt["upsample_mode"])
        noise = torch.randn(frame.shape).to(device)
        frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
        curr_r *= r
    frame = F.interpolate(frame, size=full_size, mode=opt["upsample_mode"])
    noise = torch.randn(frame.shape).to(device)
    frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
    print(frame.shape)
    return frame

def save_models(generators, discriminators, opt):
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

def train_single_scale(generators, discriminators, opt):
            
    # Initialize the dataset
    dataset = Dataset(os.path.join(input_folder, opt["training_folder"]), opt)

    torch.manual_seed(0)
    
    # Move all models to this GPU and make them distributed
    for i in range(len(generators)):
        generators[i].cuda(opt["device"])
        generators[i].eval()
        for param in generators[i].parameters():
            param.requires_grad = False
    
    # Create the new generator and discriminator for this level
    generator, num_kernels_this_scale = init_gen(len(generators), opt)
    generator = generator.cuda(opt["device"])
    discriminator = init_discrim(len(generators), opt).cuda(opt["device"])

    print_to_log_and_console(generator, os.path.join(opt["save_folder"], opt["save_name"]),
        "log.txt")
    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    print_to_log_and_console("Kernels this scale: %i" % num_kernels_this_scale, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    generator_optimizer = optim.Adam(generator.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
 
    writer = SummaryWriter(os.path.join('tensorboard',
    "%.02fdownscaleratio_physicalconstraints=%s" % 
    (opt["spatial_downscale_ratio"], opt["physical_constraints"])))

    start_time = time.time()
    next_save = 0
    images_seen = 0

    # Get properly sized frame for this generator
    full_res = dataset.__getitem__(0)
    full_res = full_res.cuda(non_blocking=True)
    
    print(str(len(generators)) + ": " + str(opt["resolutions"][len(generators)]))
    real = F.interpolate(full_res, size=opt["resolutions"][len(generators)],
    mode=opt["downsample_mode"])

    optimal_LR = torch.zeros(generator.get_input_shape(), device=opt["device"])
    opt["noise_amplitudes"].append(1.0)
    if(len(generators) > 0):
        optimal_LR = generate(generators, "reconstruct", opt, opt["device"])
        optimal_LR = F.interpolate(optimal_LR, size=opt["resolutions"][len(generators)],
        mode=opt["upsample_mode"])
        rmse = torch.sqrt(torch.mean((optimal_LR - real)**2))
        opt["noise_amplitudes"][-1] = rmse.item()
    
    writer.add_graph(generator, [optimal_LR, generator.optimal_noise])

    for epoch in range(opt["epochs"]):        
        # Generate fake image
        if(len(generators) > 0):
            fake = generate(generators, "random", opt, opt["device"])
            fake = F.interpolate(fake, size=opt["resolutions"][len(generators)],
            mode=opt["upsample_mode"])
        else:
            fake = torch.zeros(generator.get_input_shape()).to(opt["device"])
        fake = generator(fake, 
        opt["noise_amplitudes"][-1] * torch.randn(generator.get_input_shape()).to(opt["device"]))
        
        # Update discriminator: maximize D(x) + D(G(z))
        if(opt["alpha_2"] > 0.0):            
            for j in range(opt["discriminator_steps"]):
                D_loss = 0
                # Train with real downscaled to this scale
                discriminator.zero_grad()
                output = discriminator(real)
                D_loss += output.mean().item()
                discrim_error_real = opt["alpha_2"] * -output.mean()
                discrim_error_real.backward(retain_graph=True)

                # Train with the generated image
                output = discriminator(fake.detach())
                D_loss -= output.mean().item()
                discrim_error_fake = opt["alpha_2"] * output.mean()
                discrim_error_fake.backward(retain_graph=True)

                discriminator_optimizer.step()

        # Update generator: maximize D(G(z))
        for j in range(opt["generator_steps"]):
            generator.zero_grad()
            if opt["alpha_2"] > 0.0:
                output = discriminator(fake.detach())
                G_loss = output.mean().item()
                generator_error = -output.mean() * opt["alpha_2"]
                generator_error.backward(retain_graph=True)
            
            loss = nn.MSELoss().cuda(opt["device"])
            
            # Re-compute the constructed image
            optimal_reconstruction = generator(optimal_LR.detach().clone(), 
            opt["noise_amplitudes"][-1]*generator.optimal_noise)

            g = TAD(optimal_reconstruction, opt["device"])            

            if(opt["physical_constraints"] == "soft"):
                phys_loss = opt["alpha_3"] * g 
                phys_loss.backward(retain_graph = True)

            rec_loss = opt["alpha_1"]*loss(optimal_reconstruction, real)
            rec_loss.backward(retain_graph=True)
            generator_optimizer.step()
        
        writer.add_scalar('D_loss_scale%i'%len(generators), D_loss, epoch) 
        writer.add_scalar('G_loss_scale%i'%len(generators), G_loss, epoch) 
        writer.add_scalar('Rec_loss_scale%i'%len(generators), rec_loss, epoch) 
        writer.add_scalar('TAD_scale%i'%len(generators), g, epoch)

    generator = reset_grads(generator, False)
    generator.eval()
    discriminator = reset_grads(discriminator, False)
    discriminator.eval()

    return generator, discriminator

class SinGAN_Generator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, kernel_size,
    stride, pre_padding, mode, physical_constraints, device):
        super(SinGAN_Generator, self).__init__()
        modules = []
        self.pre_padding = pre_padding
        self.resolution = resolution
        self.num_channels = num_channels
        self.optimal_noise = torch.randn(self.get_input_shape(), device=device)
        self.physical_constraints = physical_constraints
        self.device = device
        if(physical_constraints and mode == "2D"):
            output_chans = 1
        else:
            output_chans = num_channels

        if(pre_padding):
            pad_amount = int(kernel_size/2)
            self.layer_padding = 0
        else:
            pad_amount = 0
            self.layer_padding = 1

        if(mode == "2D"):
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

        for i in range(num_blocks):
            # The head goes from numChannels channels to numKernels
            if i == 0:
                modules.append(nn.Sequential(
                    conv_layer(num_channels, num_kernels, kernel_size=kernel_size, 
                    stride=stride, padding=self.layer_padding),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from kernel_size to num_channels before tanh [-1,1]
            elif i == num_blocks-1:  
                tail = nn.Sequential(
                    conv_layer(num_kernels, output_chans, kernel_size=kernel_size, 
                    stride=stride, padding=self.layer_padding),
                    nn.Tanh()
                )              
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    conv_layer(num_kernels, num_kernels, kernel_size=kernel_size,
                    stride=stride, padding=self.layer_padding),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model = nn.Sequential(*modules)
        self.model = self.model.to(device)
    
    def get_input_shape(self):
        shape = []
        shape.append(1)
        shape.append(self.num_channels)
        for i in range(len(self.resolution)):
            shape.append(self.resolution[i])
        return shape

    def forward(self, data, noise):
        noisePlusData = data.clone() + noise
        if(self.pre_padding):
            noisePlusData = F.pad(noisePlusData, self.required_padding)

        if(self.physical_constraints == "hard"):
            output =  self.model(noisePlusData)
            x = -spatial_derivative2D(output, "y", self.device)
            y = spatial_derivative2D(output, "x", self.device)
            output = torch.cat((x, y), 1)
            return output
        else:
            residual = self.model(noisePlusData)
            return residual + data

class SinGAN_Discriminator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, 
    kernel_size, stride, use_sn, mode, device):
        super(SinGAN_Discriminator, self).__init__()
        self.device=device
        modules = []
        self.resolution = resolution

        if(mode == "2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(mode == "3D"):
            conv_layer = nn.Conv3d
            batchnorm_layer = nn.BatchNorm3d

        for i in range(num_blocks):
            # The head goes from 3 channels (RGB) to num_kernels
            if i == 0:
                modules.append(nn.Sequential(
                    conv_layer(num_channels, num_kernels, 
                    kernel_size, stride, 0),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            # The tail will go from num_kernels to 1 channel for discriminator optimization
            elif i == num_blocks-1:  
                tail = nn.Sequential(
                    conv_layer(num_kernels, 1, 
                    kernel_size, stride, 0)
                )
                modules.append(tail)
            # Other layers will have 32 channels for the 32 kernels
            else:
                modules.append(nn.Sequential(
                    conv_layer(num_kernels, num_kernels, 
                    kernel_size, stride, 0),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
        self.model =  nn.Sequential(*modules)
        self.model = self.model.to(device)
        if(use_sn):
            self.model = torch.nn.utils.spectral_norm(self.model)

    def forward(self, x):
        return self.model(x)

def create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn):
    bnl = batchnorm_layer(num_kernels)
    if(use_sn):
        bnl = torch.nn.utils.spectral_norm(bnl)
    return bnl

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_location, opt=None):
    self.dataset_location = dataset_location
    self.channel_mins = []
    self.channel_maxs = []
    self.num_training_examples = 1

  def __len__(self):
    return 1

  def __getitem__(self, index):
    data = np.load(os.path.join(self.dataset_location, str(0) + "_256x256.npy"))

    data = np2torch(data, "cpu")
    return data.unsqueeze(0)