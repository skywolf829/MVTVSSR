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
    return g

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
        factor =  1 / (1 / opt["spatial_downscale_ratio"])**(opt["n"] - i - 1)
        for j in range(len(dataset.resolution)):
            x = math.ceil(dataset.resolution[j] * factor)
            scaling.append(x)
        #opt["resolutions"].insert(0, scaling)
        opt["resolutions"].append(scaling)
        print("Scale %i: %s" % (opt["n"] - 1 - i, str(scaling)))

def init_gen(scale, opt):
    num_kernels = int( 2** ((math.log(opt["base_num_kernels"]) / math.log(2)) + (scale / 6)))

    generator = SinGAN_Generator(opt["resolutions"][scale], opt["num_blocks"], 
    opt["num_channels"], num_kernels, opt["kernel_size"], opt["stride"], 
    opt["pre_padding"], opt["mode"], opt["physical_constraints"], opt['separate_chans'], scale,
    opt["device"])

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
        noise = torch.zeros(frame.shape).to(device)
        frame = generator(frame, opt["noise_amplitudes"][-1]*noise)
        curr_r *= r
    frame = F.interpolate(frame, size=full_size, mode=opt["upsample_mode"])
    noise = torch.randn(frame.shape).to(device)
    noise = torch.zeros(frame.shape).to(device)
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
        generators[i].to(opt["device"])
        generators[i].eval()
        for param in generators[i].parameters():
            param.requires_grad = False
    
    # Create the new generator and discriminator for this level
    generator, num_kernels_this_scale = init_gen(len(generators), opt)
    generator = generator.to(opt["device"])
    discriminator = init_discrim(len(generators), opt).to(opt["device"])

    print_to_log_and_console(generator, os.path.join(opt["save_folder"], opt["save_name"]),
        "log.txt")
    print_to_log_and_console("Training on %s" % (opt["device"]), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    print_to_log_and_console("Kernels this scale: %i" % num_kernels_this_scale, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")


    generator_optimizer = optim.Adam(generator.get_params(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,
    milestones=[1600],gamma=opt['gamma'])
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=opt["learning_rate"], 
    betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_optimizer,
    milestones=[1600],gamma=opt['gamma'])
 
    writer = SummaryWriter(os.path.join('tensorboard',
    "%.02fsdr_pc=%s_a1=%0.02f_a2=%0.02f_a3=%0.02f_a4=%0.02f_lr=%0.06f_sn=%s" % 
    (opt["spatial_downscale_ratio"], opt["physical_constraints"], opt['alpha_1'], opt['alpha_2'], 
    opt['alpha_3'], opt['alpha_4'], opt['learning_rate'], opt["use_spectral_norm"])))

    start_time = time.time()
    next_save = 0
    images_seen = 0

    # Get properly sized frame for this generator
    full_res = dataset.__getitem__(0)
    full_res = full_res.to(opt["device"])
    
    print(str(len(generators)) + ": " + str(opt["resolutions"][len(generators)]))
    
    real = full_res.clone()
    if(len(generators) + 1 is not len(opt["resolutions"])):
        real = pyramid_reduce(full_res[0].clone().cpu().numpy().swapaxes(0,2).swapaxes(0,1), 
        downscale = (1 / opt["spatial_downscale_ratio"])**(len(opt["resolutions"]) - len(generators)-1),
        multichannel=True).swapaxes(0,1).swapaxes(0,2)
        real = np2torch(real, device=opt["device"]).unsqueeze(0)
    

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
            fake_prev = generate(generators, "random", opt, opt["device"])
            fake_prev = F.interpolate(fake_prev, size=opt["resolutions"][len(generators)],
            mode=opt["upsample_mode"])
        else:
            fake_prev = torch.zeros(generator.get_input_shape()).to(opt["device"])
        
        D_loss = 0
        G_loss = 0
        
        # Update discriminator: maximize D(x) + D(G(z))
        if(opt["alpha_2"] > 0.0):            
            for j in range(opt["discriminator_steps"]):
                noise = opt["noise_amplitudes"][-1] * torch.randn(generator.get_input_shape()).to(opt["device"])
                fake = generator(fake_prev.detach(), noise)
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
                noise = opt["noise_amplitudes"][-1] * torch.randn(generator.get_input_shape()).to(opt["device"])
                fake = generator(fake_prev.detach(), noise)
                output = discriminator(fake)
                generator_error = -output.mean() * opt["alpha_2"]
                generator_error.backward(retain_graph=True)
                G_loss = output.mean().item()


            loss = nn.L1Loss().cuda(opt["device"])
            
            # Re-compute the constructed image
            optimal_reconstruction = generator(optimal_LR.detach().clone(), 
            opt["noise_amplitudes"][-1]*generator.optimal_noise)
            g = 0
            if(real.shape[1] > 1):
                g_map = TAD(dataset.unscale(optimal_reconstruction), opt["device"])            
                g = g_map.sum()
            mags = np.zeros(1)
            angles = np.zeros(1)
            if(real.shape[1] > 1):
                cs = torch.nn.CosineSimilarity(dim=1)
                #mags = mag_difference(optimal_reconstruction, real)
                mags = torch.abs(torch.norm(optimal_reconstruction, dim=1) - torch.norm(real, dim=1))
                angles = torch.abs(cs(optimal_reconstruction, real) - 1) / 2

            if(opt["physical_constraints"] == "soft"):
                phys_loss = opt["alpha_3"] * g 
                phys_loss.backward(retain_graph = True)
            
            rec_loss = loss(optimal_reconstruction, real)
            
            if(opt['alpha_1'] > 0.0):
                rec_loss *= opt["alpha_1"]
                rec_loss.backward(retain_graph=True)
            if(opt['alpha_4'] > 0.0):
                cs = torch.nn.CosineSimilarity(dim=1)                
                r_loss = opt['alpha_4'] * (mags.mean() + angles.mean()) / 2
                r_loss.backward(retain_graph=True)
            generator_optimizer.step()

        if epoch == 0:
            noise_numpy = opt["noise_amplitudes"][-1]*generator.optimal_noise.clone().detach().cpu().numpy()[0]
            noise_cm = toImg(noise_numpy)
            writer.add_image("noise/%i"%len(generators), 
            noise_cm, epoch)            
            real_numpy = real.clone().detach().cpu().numpy()[0]
            real_cm = toImg(real_numpy)
            writer.add_image("real/%i"%len(generators), 
            real_cm, epoch)

            writer.add_image("realx/%i"%len(generators), 
            toImg(real_numpy[0:1]), epoch)
            writer.add_image("realy/%i"%len(generators), 
            toImg(real_numpy[1:2]), epoch)

            

        if(epoch % 50 == 0):
            rec_numpy = optimal_reconstruction.clone().detach().cpu().numpy()[0]
            rec_cm = toImg(rec_numpy)
            writer.add_image("reconstructed/%i"%len(generators), 
            rec_cm, epoch)
            writer.add_image("reconstructedx/%i"%len(generators), 
            toImg(rec_numpy[0:1]), epoch)
            writer.add_image("reconstructedy/%i"%len(generators), 
            toImg(rec_numpy[1:2]), epoch)

            if(opt["alpha_2"] > 0.0):
                fake_numpy = fake.clone().detach().cpu().numpy()[0]
                fake_cm = toImg(fake_numpy)
                writer.add_image("fake/%i"%len(generators), 
                fake_cm, epoch)
            if(real.shape[1] > 1):
                angles_cm = toImg(angles.detach().cpu().numpy())
                writer.add_image("angle/%i"%len(generators), 
                angles_cm , epoch)

                mags_cm = toImg(mags.detach().cpu().numpy())
                writer.add_image("mag/%i"%len(generators), 
                mags_cm, epoch)

                g_cm = toImg(g_map.detach().cpu().numpy()[0])
                writer.add_image("Divergence/%i"%len(generators), 
                g_cm, epoch)
        
        print_to_log_and_console("%i/%i: Dloss=%.02f Gloss=%.02f L1=%.04f TAD=%.02f AMD=%.02f AAD=%.02f" %
        (epoch, opt['epochs'], D_loss, G_loss, rec_loss, g, mags.mean(), angles.mean()), 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

        writer.add_scalar('D_loss_scale/%i'%len(generators), D_loss, epoch) 
        writer.add_scalar('G_loss_scale/%i'%len(generators), G_loss, epoch) 
        writer.add_scalar('L1/%i'%len(generators), rec_loss, epoch) 
        writer.add_scalar('TAD_scale/%i'%len(generators), g, epoch)
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
    stride, pre_padding, mode, physical_constraints, separate_chans, scale, device):
        super(SinGAN_Generator, self).__init__()
        self.scale = scale
        self.pre_padding = pre_padding
        self.resolution = resolution
        self.num_channels = num_channels
        print(self.scale)
        if(self.scale == 0):
            self.optimal_noise = torch.randn(self.get_input_shape(), device=device)
        else:
            self.optimal_noise = torch.zeros(self.get_input_shape(), device=device)
        self.physical_constraints = physical_constraints
        self.device = device
        self.separate_chans = separate_chans

        if(physical_constraints == "hard" and mode == "2D"):
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
                    stride=stride, padding=self.layer_padding, groups=groups)
                    #nn.Tanh()
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

    def forward(self, data, noise=None):
        if(noise is None):
            noise = torch.zeros(data.shape).to(self.device)
        noisePlusData = data.clone() + noise
        if(self.pre_padding):
            noisePlusData = F.pad(noisePlusData, self.required_padding)

        output = self.model(noisePlusData)

        if(self.physical_constraints == "hard"):
            x = -spatial_derivative2D(output, "y", self.device)
            y = spatial_derivative2D(output, "x", self.device)
            output = torch.cat((x, y), 1)
            return output
        else:
            return output + data

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
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname.find('Conv2d') != -1:
                    m = torch.nn.utils.spectral_norm(m)
                elif classname.find('Norm') != -1:
                    m = torch.nn.utils.spectral_norm(m)

    def forward(self, x):
        return self.model(x)

def create_batchnorm_layer(batchnorm_layer, num_kernels, use_sn):
    bnl = batchnorm_layer(num_kernels)
    if(use_sn):
        bnl = torch.nn.utils.spectral_norm(bnl)
    return bnl

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
        for filename in os.listdir(self.dataset_location):
            self.num_items += 1
            d = np.load(os.path.join(self.dataset_location, filename))
            self.num_channels = d.shape[0]
            self.resolution = d.shape[1:]
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

    def unscale(self, data):
        d = data.clone()
        if(self.scale_data):
            for i in range(self.num_channels):
                d[0, i] *= 0.5
                d[0, i] += 0.5
                d[0, i] *= (self.channel_maxs[i] - self.channel_mins[i])
                d[0, i] += self.channel_mins[i]
        elif(self.scale_on_magnitude):
            data *= self.max_mag
        return d

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

        data = np2torch(data, "cpu")
        return data.unsqueeze(0)