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

def calc_gradient_penalty(discriminator, real_data, fake_data, latent_vector, LAMBDA, device):
    alpha = torch.rand(1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates, latent_vector.detach())

    grad_outputs=torch.ones(disc_interpolates.size()).cuda(device)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def generate_padded_noise(size, pad_size, pad_with_noise, device):
    if(pad_with_noise):
        for i in range(2,len(size)):
            size[i] += 2*pad_size
        noise = torch.randn(size, device=device)
    else:
        noise = torch.randn(size, device=device)
        pad_zero = nn.ZeroPad2d(pad_size)
        noise = pad_zero(noise)
    return noise

def init_scales(opt):
    ns = []
    if(opt["downscale_ratio"] < 1.0):
        ns.append(round(math.log(opt["min_dimension_size"] / opt["base_resolution"][0]) / math.log(opt["downscale_ratio"])) + 1)
        ns.append(round(math.log(opt["min_dimension_size"] / opt["base_resolution"][1]) / math.log(opt["downscale_ratio"])) + 1)
    opt["n"] = min(ns)
    print("The model will have %i scales" % (opt["n"]))
    for i in range(opt["n"]):
        x = round(opt["base_resolution"][1] * (opt["downscale_ratio"] ** i))
        y = round(opt["base_resolution"][0] * (opt["downscale_ratio"] ** i))
        scaling = [y, x]
        opt["scales"].insert(0, scaling)
        print("Scale %i: %s" % (opt["n"] - 1 - i, str(scaling)))

def init_gen(scale, opt):
    num_kernels_exp = math.log(opt["base_num_kernels"]) / math.log(2)
    num_kernels = int(math.pow(2, num_kernels_exp + ((opt["n"]-scale) / 1.25)))

    generator = MVTVSSR_Generator(opt["scales"][scale], opt["num_blocks"], opt["num_channels"],
    num_kernels, opt["kernel_size"], opt["stride"], 
    opt["conv_layer_padding"], opt["network_input_padding"], opt["mode"])
    weights_init(generator)
    return generator

def init_discrim_s(scale, opt):
    num_kernels_exp = math.log(opt["base_num_kernels"]) / math.log(2)
    num_kernels = int(math.pow(2, num_kernels_exp + ((opt["n"]-scale) / 1.25)))

    discriminator = MVTVSSR_Spatial_Discriminator(opt["scales"][scale], opt["num_blocks"], opt["num_channels"],
    num_kernels, opt["kernel_size"], opt["stride"], 
    opt["conv_layer_padding"], opt["use_spectral_norm"], opt["mode"])
    weights_init(discriminator)
    return discriminator

def init_discrim_t(scale, opt):
    num_kernels_exp = math.log(opt["base_num_kernels"]) / math.log(2)
    num_kernels = int(math.pow(2, num_kernels_exp + ((opt["n"]-scale) / 1.25)))

    discriminator = MVTVSSR_Temporal_Discriminator(opt["scales"][scale], opt["num_blocks"], opt["num_channels"],
    num_kernels, opt["kernel_size"], opt["stride"], 
    opt["conv_layer_padding"], opt["use_spectral_norm"], opt["mode"])
    weights_init(discriminator)
    return discriminator

def generate(generators, opt, minibatch, mode, starting_frames, device):

    generated_image = starting_frames

    for i in range(len(generators)):
        generator = generators[i]
        shape = list(generated_image.shape)
        if(mode == "random"):
            noise = opt["noise_amplitudes"][i] * generate_padded_noise(shape, 0, False, device)
        else:
            noise = torch.zeros(shape, device=device)
        generated_image = generator(generated_image, noise, opt["mode"])
        if(i < len(generators) - 1):
            generated_image = F.interpolate(generated_image, 
            size=opt["scales"][i+2], mode=opt["upsample_mode"], align_corners=True)
    return generated_image

def save_models(generators, discriminators_s, discriminators_t, opt):
    folder = create_folder(opt["save_folder"], opt["save_name"])
    path_to_save = os.path.join(opt["save_folder"], folder)
    print_to_log_and_console("Saving model to %s" % (path_to_save), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if(opt["save_generators"]):
        gen_states = {}
        for i in range(len(generators)):
            gen_states[str(i)] = generators[i].state_dict()
        torch.save(gen_states, os.path.join(path_to_save, "MVTVSSRGAN.generators"))
    if(opt["save_discriminators_s"]):
        discrim_states = {}
        for i in range(len(discriminators_s)):
            discrim_states[str(i)] = discriminators_s[i].state_dict()
        torch.save(discrim_states, os.path.join(path_to_save, "MVTVSSRGAN.s_discriminators"))
    if(opt["save_discriminators_t"]):
        discrim_states = {}
        for i in range(len(discriminators_t)):
            discrim_states[str(i)] = discriminators_t[i].state_dict()
        torch.save(discrim_states, os.path.join(path_to_save, "MVTVSSRGAN.t_discriminators"))

    save_options(opt, path_to_save)

def load_models(opt, device):
    generators = []
    discriminators_s = []
    discriminators_t = []
    load_folder = os.path.join(opt["save_folder"], opt["save_name"])

    if not os.path.exists(load_folder):
        print_to_log_and_console("%s doesn't exist, load failed" % load_folder, 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(load_folder, "MVTVSSRGAN.generators")):
        gen_params = torch.load(os.path.join(load_folder, "MVTVSSRGAN.generators"),map_location=device)
        for i in range(opt["n"]-1):
            if(str(i) in gen_params.keys()):
                gen_params_compat = OrderedDict()
                for k, v in gen_params[str(i)].items():
                    if("module" in k):
                        gen_params_compat[k[7:]] = v
                    else:
                        gen_params_compat[k] = v
                generator = init_gen(i+1, opt)
                generator.load_state_dict(gen_params_compat)
                generators.append(generator)
        print_to_log_and_console("Successfully loaded MVTVSSRGAN.generators", os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "MVTVSSRGAN.generators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if os.path.exists(os.path.join(load_folder, "MVTVSSRGAN.s_discriminators")):
        discrim_params = torch.load(os.path.join(load_folder, "MVTVSSRGAN.s_discriminators"),map_location=device)
        for i in range(opt["n"]):
            if(str(i) in discrim_params.keys()):
                discrim_params_compat = OrderedDict()
                for k, v in discrim_params[str(i)].items():
                    if(k[0:7] == "module."):
                        discrim_params_compat[k[7:]] = v
                    else:
                        discrim_params_compat[k] = v
                discriminator = init_discrim_s(i+1, opt)
                discriminator.load_state_dict(discrim_params_compat)
                discriminators_s.append(discriminator)
        print_to_log_and_console("Successfully loaded MVTVSSRGAN.s_discriminators", os.path.join(opt["save_folder"],opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "MVTVSSRGAN.s_discriminators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    if os.path.exists(os.path.join(load_folder, "MVTVSSRGAN.t_discriminators")):
        discrim_params = torch.load(os.path.join(load_folder, "MVTVSSRGAN.t_discriminators"),map_location=device)
        for i in range(opt["n"]):
            if(str(i) in discrim_params.keys()):
                discrim_params_compat = OrderedDict()
                for k, v in discrim_params[str(i)].items():
                    if(k[0:7] == "module."):
                        discrim_params_compat[k[7:]] = v
                    else:
                        discrim_params_compat[k] = v
                discriminator = init_discrim_t(i+1, opt)
                discriminator.load_state_dict(discrim_params_compat)
                discriminators_t.append(discriminator)
        print_to_log_and_console("Successfully loaded MVTVSSRGAN.t_discriminators", os.path.join(opt["save_folder"],opt["save_name"]), "log.txt")
    else:
        print_to_log_and_console("Warning: %s doesn't exists - can't load these model parameters" % "MVTVSSRGAN.t_discriminators", 
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    
    return  generators, discriminators_s, discriminators_t

def save_training_graph(losses, scale_num, opt):
    fig, ax = plt.subplots(4, gridspec_kw={'width_ratios': [0.5]})
    fig.subplots_adjust(left=0.125, right=0.95, bottom=0.10, top=0.95, wspace=0.2, hspace=0.90)
    

    ax[0].plot(range(0, len(losses[0])), losses[0])
    ax[0].set_xlim(0, len(losses[0]))
    ax[0].set_ylim(min(losses[0]), max(losses[0]))
    ax[0].set_title("Discriminator Loss, Real - Scale %i" % (scale_num))
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(range(0, len(losses[1])), losses[1])
    ax[1].set_xlim(0, len(losses[1]))
    ax[1].set_ylim(min(losses[1]), max(losses[1]))
    ax[1].set_title("Discriminator Loss, Fake - Scale %i" % (scale_num))
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Epoch")

    ax[2].plot(range(0, len(losses[2])), losses[2])
    ax[2].set_xlim(0, len(losses[2]))
    ax[2].set_ylim(min(losses[2]), max(losses[2]))
    ax[2].set_title("Generator Loss - Scale %i" % (scale_num))
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Epoch")

    ax[3].plot(range(0, len(losses[3])), losses[3])
    ax[3].set_xlim(0, len(losses[3]))
    ax[3].set_ylim(min(losses[3]), max(losses[3]))
    ax[3].set_title("Reconstruction Error - Scale %i" % (scale_num))
    ax[3].set_ylabel("Error")
    ax[3].set_xlabel("Epoch")

    plt.savefig(os.path.join(opt["save_folder"], opt["save_name"], "errs_"+str(scale_num)+".png"))
    plt.close()

def train_single_scale(process_num, generators, discriminators_s, discriminators_t, opt):
    # Distributed - torch will call this with a different process_num (0, ... N-1)
    # Set up distributed environment
    # Locking call - syncs up processes

    if(opt["train_distributed"]):
        dist.init_process_group(backend='nccl',
            world_size=len(opt["device"]),
            rank=process_num
        )
        gpu_id = opt["device"][0]
        torch.cuda.set_device(gpu_id)
    else:
        gpu_id = opt["device"][process_num]
        torch.cuda.set_device(gpu_id)
        
    losses = [[],[],[],[]]
    torch.manual_seed(0)
    
    print_to_log_and_console("Process num %i training on GPU %i" % (process_num, gpu_id), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

    # Move all models to this GPU and make them distributed
    for i in range(len(generators)):
        generators[i].cuda(gpu_id)
        generators[i].eval()
        for param in generators[i].parameters():
            param.requires_grad = False
    
    # Create the new generator and discriminator for this level
    generator = init_gen(len(generators)+1, opt).cuda(gpu_id)    
    discriminator_s = init_discrim_s(len(generators)+1, opt).cuda(gpu_id)
    discriminator_t = init_discrim_t(len(generators)+1, opt).cuda(gpu_id)

    if(opt["train_distributed"]):
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[gpu_id], find_unused_parameters=True)
        discriminator_s = nn.parallel.DistributedDataParallel(discriminator_s, device_ids=[gpu_id], find_unused_parameters=True)
        discriminator_t = nn.parallel.DistributedDataParallel(discriminator_t, device_ids=[gpu_id], find_unused_parameters=True)

    num_kernels_exp = math.log(opt["base_num_kernels"]) / math.log(2)
    num_kernels_this_scale = int(math.pow(2, num_kernels_exp + ((opt["n"]-len(generators)) / 2.0)))
    if(process_num == 0):
        print_to_log_and_console("Kernels this scale: %i" % num_kernels_this_scale, 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    num_kernels_factor = num_kernels_this_scale / opt["base_num_kernels"]

    # Adjust minibatch based on scale size and number of kernels at this scale
    minibatch_this_scale = int((opt["minibatch"] * (opt["scales"][-1][0] / opt["scales"][len(generators)][0]) * (opt["scales"][-1][1]  / opt["scales"][len(generators)][1])) / num_kernels_factor)
    '''
    if(process_num == 0):
        print_to_log_and_console(generator, 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        print_to_log_and_console(discriminator_s, 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        print_to_log_and_console(discriminator_t, 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
        print_to_log_and_console("Minibatch size this scale: %i" % (minibatch_this_scale), 
            os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    '''
    # Initialize the dataset
    dataset = Dataset(os.path.join(input_folder, opt["data_folder"]), opt["num_training_examples"])

    if(opt["train_distributed"]):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=len(opt["device"]),
            rank=process_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=minibatch_this_scale,
            shuffle=False,
            num_workers=opt["num_workers"],
            sampler=train_sampler
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=minibatch_this_scale,
            shuffle=train_single_scale,
            num_workers=opt["num_workers"]
        )

    total_minibatches = opt["num_training_examples"] / (len(opt["device"]) * minibatch_this_scale)

    generator_optimizer = optim.Adam(generator.parameters(), lr=opt["learning_rate"], betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_s_optimizer = optim.Adam(discriminator_s.parameters(), lr=opt["learning_rate"], betas=(opt["beta_1"],opt["beta_2"]))
    discriminator_t_optimizer = optim.Adam(discriminator_t.parameters(), lr=opt["learning_rate"], betas=(opt["beta_1"],opt["beta_2"]))
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=generator_optimizer,milestones=[1600],gamma=opt["gamma"])
    discriminator_s_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_s_optimizer,milestones=[1600],gamma=opt["gamma"])
    discriminator_t_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=discriminator_t_optimizer,milestones=[1600],gamma=opt["gamma"])
    
    start_time = time.time()
    images_seen = 0
    for epoch in range(opt["iteration_number"], opt["epochs"]):
        for i, (output_dataframes) in enumerate(dataloader):

            # Get our minibatch on the gpu
            minibatch_size = output_dataframes.shape[0]
            output_dataframes = output_dataframes.cuda(non_blocking=True)
            output_dataframes_this_scale = F.interpolate(output_dataframes.clone(), size=opt["scales"][len(generators)+1], mode=opt["downsample_mode"], align_corners=True)
            
            
            # Calculate the noise amplitude needed at this level            
            if(epoch == 0 and i == 0):
                criterion = nn.MSELoss().cuda()
                lowest_res = F.interpolate(output_dataframes.clone(), size=opt["scales"][0], mode=opt["downsample_mode"], align_corners=True)
                if(len(generators) == 0):
                    reconstructed_frames =  F.interpolate(lowest_res, size=opt["scales"][1], mode=opt["upsample_mode"], align_corners=True)
                    noise_amp = torch.sqrt(criterion(reconstructed_frames, output_dataframes_this_scale)).item()
                elif(len(generators) > 0):
                    reconstructed_frames = generate(generators, opt, minibatch_size, "reconstruct", lowest_res, "cuda:"+str(gpu_id))
                    reconstructed_frames = F.interpolate(reconstructed_frames, size=opt["scales"][len(generators)+1], mode=opt["upsample_mode"], align_corners=True)
                    noise_amp = torch.sqrt(criterion(reconstructed_frames, output_dataframes_this_scale)).item()
                opt["noise_amplitudes"].append(noise_amp)


            # Downsample it to the min scale
            downscaled_dataframes = F.interpolate(output_dataframes.clone(), size=opt["scales"][0], mode=opt["downsample_mode"], align_corners=True)
            # Upsample that to the next scale
            generated_fake = F.interpolate(downscaled_dataframes, size=opt["scales"][1], mode=opt["upsample_mode"], align_corners=True)
            
            # Send through other frozen generators
            if(len(generators) > 0):
                generated_fake = generate(generators, opt, minibatch_size, "random", generated_fake, "cuda:"+str(gpu_id))
                # Upscale to this scale now
                generated_fake = F.interpolate(generated_fake, size=opt["scales"][len(generators)+1], mode=opt["upsample_mode"], align_corners=True)
            
            # Get random noise scaled by our amplitude at this level
            noise = opt["noise_amplitudes"][-1]*generate_padded_noise(list(generated_fake.shape), 0, False, opt["device"][0])

            # Send input with noise through the current generator in training 
            generated_fake = generator(generated_fake, noise, opt["mode"])
            
            # Update discriminator: maximize D(x) + D(G(z))
            for j in range(opt["discriminator_steps"]):

                # Train with real downscaled to this scale
                discriminator_s.zero_grad()
                output = discriminator_s(output_dataframes_this_scale)
                discrim_error_real = -output.mean()
                discrim_error_real.backward(retain_graph=True)
                losses[0].append(discrim_error_real.detach().item())

                # Train with the fake generated image
                output = discriminator_s(generated_fake.detach())
                discrim_error_fake = output.mean()
                discrim_error_fake.backward(retain_graph=True)
                losses[1].append(discrim_error_fake.detach().item())

                discriminator_s_optimizer.step()

            # Update generator: maximize D(G(z))
            for j in range(opt["generator_steps"]):
                generator.zero_grad()
                output = discriminator_s(generated_fake.detach())
                generator_error = -output.mean()
                generator_error.backward(retain_graph=True)
                losses[2].append(discrim_error_fake.detach().item())

                loss = nn.MSELoss().cuda(gpu_id)
                
                # Generate reconstructed frames for this level
                lowest_res = F.interpolate(output_dataframes.clone(), size=opt["scales"][0], mode=opt["downsample_mode"], align_corners=True)
                if(len(generators) == 0):
                    reconstructed_frames =  F.interpolate(lowest_res, size=opt["scales"][1], mode=opt["upsample_mode"], align_corners=True)
                elif(len(generators) > 0):
                    reconstructed_frames = generate(generators, opt, minibatch_size, "reconstruct", lowest_res, "cuda:"+str(gpu_id))
                    reconstructed_frames = F.interpolate(reconstructed_frames, size=opt["scales"][len(generators)+1], mode=opt["upsample_mode"], align_corners=True)
                reconstructed_frames = generator(reconstructed_frames, torch.zeros(reconstructed_frames.shape, device=opt["device"][0]),opt["mode"])

                rec_loss = opt["alpha"]*loss(reconstructed_frames, output_dataframes_this_scale)
                rec_loss.backward(retain_graph=True)
                losses[3].append(rec_loss.detach().item())
                generator_optimizer.step()

            images_seen += minibatch_this_scale * len(opt["device"])
            print_to_log_and_console("Level %i epoch [%i/%i] iteration[%i/%i]: %.04f images/sec" % (len(generators),epoch, opt["epochs"], i, total_minibatches, images_seen / (time.time() - start_time)), 
                os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
            print_to_log_and_console("Average discrim_real: %.04f discrim_fake: %.04f gen_err %.04f rec_err %.04f" % ((np.array(losses[0])).mean() / images_seen, 
            (np.array(losses[1])).mean() / images_seen, (np.array(losses[2])).mean() / images_seen, (np.array(losses[3])).mean() / images_seen), 
                os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

        discriminator_s_scheduler.step()
        generator_scheduler.step()

    generator = reset_grads(generator, False)
    generator.eval()
    discriminator_s = reset_grads(discriminator_s, False)
    discriminator_s.eval()

    save_training_graph(losses, len(generators), opt)

    if(process_num == 0 and opt["train_distributed"]):
        generators.append(generator)
        discriminators_s.append(discriminator_s)
        discriminators_t.append(discriminator_t)
        save_models(generators, discriminators_s, discriminators_t, opt)
    else:
        return generator, discriminator_s, discriminator_t

class MVTVSSR_Generator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, kernel_size, 
    stride, layer_padding, pre_padding, mode):
        super(MVTVSSR_Generator, self).__init__()
        self.resolution = resolution
        self.model = []

        if(mode == "2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            self.padder = nn.ZeroPad2d(pre_padding)
        elif(mode == "3D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm3d
            self.padder = nn.ZeroPad3d(pre_padding)

        self.model.append(nn.Sequential(
                conv_layer(num_channels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                batchnorm_layer(num_kernels),
                nn.LeakyReLU(0.2, inplace=True)))
        for i in range(num_blocks):
            self.model.append(nn.Sequential(
                conv_layer(num_kernels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                batchnorm_layer(num_kernels),
                nn.LeakyReLU(0.2, inplace=True)))

        self.model.append(nn.Sequential(
                conv_layer(num_kernels, num_channels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                nn.Tanh()))

        self.model = nn.Sequential(*self.model)
    
    def forward(self, dataframe, noise, mode):
        dataframe_plus_noise = dataframe + noise
        padded_input = self.padder(dataframe_plus_noise)
        residual = self.model(padded_input)
        out = dataframe + residual
        return out

class MVTVSSR_Spatial_Discriminator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, kernel_size, 
    stride, layer_padding, use_spectral_norm, mode):
        super(MVTVSSR_Spatial_Discriminator, self).__init__()

        if(mode == "2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(mode == "3D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm3d

        self.resolution = resolution
        self.model = []

        if(use_spectral_norm):
            self.model.append(
                nn.Sequential(
                    SpectralNorm(conv_layer(num_channels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding)),
                    SpectralNorm(batchnorm_layer(num_kernels)),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            for i in range(num_blocks):
                self.model.append(nn.Sequential(
                    SpectralNorm(conv_layer(num_kernels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding)),
                    SpectralNorm(batchnorm_layer(num_kernels)),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            self.model.append(SpectralNorm(conv_layer(num_kernels, 1, kernel_size=kernel_size, stride=stride, padding=layer_padding)))
        else:
            self.model.append(
                nn.Sequential(
                    conv_layer(num_channels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

            for i in range(num_blocks):
                self.model.append(nn.Sequential(
                    conv_layer(num_kernels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))

            self.model.append(conv_layer(num_kernels, 1, kernel_size=kernel_size, stride=stride, padding=layer_padding))

        self.model = nn.Sequential(*self.model)

    def forward(self, dataframe):
        out = self.model(dataframe)
        return out

class MVTVSSR_Temporal_Discriminator(nn.Module):
    def __init__ (self, resolution, num_blocks, num_channels, num_kernels, kernel_size, 
    stride, layer_padding, use_spectral_norm, mode):
        super(MVTVSSR_Temporal_Discriminator, self).__init__()

        if(mode == "2D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
        elif(mode == "3D"):
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm3d

        self.resolution = resolution
        self.model = []

        if(use_spectral_norm):
            self.model.append(
                nn.Sequential(
                    SpectralNorm(conv_layer(num_channels*3, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding)),
                    SpectralNorm(batchnorm_layer(num_kernels)),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            for i in range(num_blocks):
                self.model.append(nn.Sequential(
                    SpectralNorm(conv_layer(num_kernels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding)),
                    SpectralNorm(batchnorm_layer(num_kernels)),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            self.model.append(SpectralNorm(conv_layer(num_kernels, 1, kernel_size=kernel_size, stride=stride, padding=layer_padding)))
        else:
            self.model.append(
                nn.Sequential(
                    conv_layer(num_channels*3, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

            for i in range(num_blocks):
                self.model.append(nn.Sequential(
                    conv_layer(num_kernels, num_kernels, kernel_size=kernel_size, stride=stride, padding=layer_padding),
                    batchnorm_layer(num_kernels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))

            self.model.append(conv_layer(num_kernels, 1, kernel_size=kernel_size, stride=stride, padding=layer_padding))

        self.model = nn.Sequential(*self.model)

    def forward(self, dataframe_prev, dataframe_now, dataframe_next):
        concat_datframes = torch.cat([dataframe_prev, dataframe_now, dataframe_next], dim=1)
        out = self.model(dataframe)
        return out

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_location, num_training_examples):
    self.dataset_location = dataset_location
    self.num_training_examples = num_training_examples
    self.channel_mins = []
    self.channel_maxs = []
    for index in range(num_training_examples):
        data = np.load(os.path.join(self.dataset_location, str(index) + ".npy"))
        if index == 0:
            for j in range(data.shape[0]):
                self.channel_mins.append(np.min(data[j]))
                self.channel_maxs.append(np.max(data[j]))
        else:
            for j in range(data.shape[0]):
                chan_min = np.min(data[j])
                chan_max = np.max(data[j])
                if(chan_min < self.channel_mins[j]):
                    self.channel_mins[j] = chan_min
                if(chan_max > self.channel_maxs[j]):
                    self.channel_maxs[j] = chan_max
    # If a channel is unchanging, min=max, causing a divide by 0 error later. Fix that here.
    for j in range(len(self.channel_mins)):
        if(self.channel_mins[j] == self.channel_maxs[j]):
            self.channel_maxs[j] += 1e-8
            self.channel_mins[j] -= 1e-8

  def __len__(self):
    return self.num_training_examples

  def __getitem__(self, index):
    data = np.load(os.path.join(self.dataset_location, str(index) + ".npy"))
    # Scale between [-1, 1] per channel for the dataset
    for i in range(data.shape[0]):
        data[i] -= self.channel_mins[i]
        data[i] *= (2/ (self.channel_maxs[i] - self.channel_mins[i]))
        data[i] -= 1 
    data = np2torch(data, "cpu")
    return data

# As implemented at https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
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