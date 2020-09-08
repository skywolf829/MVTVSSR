from SinGAN_models import *
from options import *
from utility_functions import *
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import os
import imageio
import argparse
import time
import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default=None,help='The type of input - 2D, 3D')
    parser.add_argument('--data_folder',default=None,type=str,help='File to train on')
    parser.add_argument('--testing_folder',default=None,type=str,help='File to validate on')
    parser.add_argument('--num_training_examples',default=None,type=int,help='Frames to use from training file')
    parser.add_argument('--save_folder',default=None, help='The folder to save the models folder into')
    parser.add_argument('--save_name',default=None, help='The name for the folder to save the model')
    parser.add_argument('--num_channels',default=None,type=int,help='Number of channels to use')
    parser.add_argument('--spatial_downscale_ratio',default=None,type=float,help='Ratio for spatial downscaling')
    parser.add_argument('--min_dimension_size',default=None,type=int,help='Minimum dimension size')

    parser.add_argument('--num_blocks',default=None,type=int, help='Num of conv-batchnorm-relu blocks per gen/discrim')
    parser.add_argument('--base_num_kernels',default=None,type=int, help='Num conv kernels in lowest layer')
    parser.add_argument('--pre_padding',default=None,type=str2bool, help='Padding before entering network')
    parser.add_argument('--kernel_size',default=None, type=int,help='Conv kernel size')
    parser.add_argument('--stride',default=None, type=int,help='Conv stride length')
    
    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--device',type=str,default=None, help='Device to use')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--ranking',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--save_generators',default=None, type=str2bool,help='Whether or not to save generators')
    parser.add_argument('--save_discriminators',default=None, type=str2bool,help='Whether or not to save discriminators')

    parser.add_argument('--alpha_1',default=None, type=float,help='Reconstruction loss coefficient')
    parser.add_argument('--alpha_2',default=None, type=float,help='Adversarial loss coefficient')
    parser.add_argument('--generator_steps',default=None, type=int,help='Number of generator steps to take')
    parser.add_argument('--discriminator_steps',default=None, type=int,help='Number of discriminator steps to take')
    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--minibatch',default=None, type=int,help='Size of minibatch to train on')
    parser.add_argument('--num_workers',default=None, type=int,help='Number of workers for dataset loader')
    parser.add_argument('--learning_rate',default=None, type=float,help='Learning rate for the networks')
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')
    parser.add_argument('--gamma',default=None, type=float,help='')
    parser.add_argument('--physical_constraints',default=None,type=str,help='none, soft, or hard')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')


    args = vars(parser.parse_args())

    MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
    output_folder = os.path.join(MVTVSSR_folder_path, "Output")
    save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")


    opt = Options.get_default()

    # Read arguments and update our options
    for k in args.keys():
        if args[k] is not None:
            opt[k] = args[k]

    # Determine scales
    init_scales(opt)

    # Init models
    generators = []
    discriminators = []
    
    now = datetime.datetime.now()
    start_time = time.time()
    print_to_log_and_console("Started training at " + str(now), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    opt["num_training_examples"] = len(os.listdir(os.path.join(input_folder,opt["training_folder"])))

    # Train each scale 1 by 1
    for i in range(opt["n"]):

        start_time_scale_n = time.time()

        print_to_log_and_console(str(datetime.datetime.now()) + " - Beginning training on scale " + str(len(generators)),
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

        generator, discriminator = train_single_scale(generators, discriminators, opt)
        generators.append(generator)
        discriminators.append(discriminator)
            
        time_passed = (time.time() - start_time_scale_n) / 60
        print_to_log_and_console("%s - Finished training in scale %i in %f minutes" % (str(datetime.datetime.now()), len(generators)-1, time_passed),
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 


    time_passed = (time.time() - start_time) / 60
    print_to_log_and_console("%s - Finished training  in %f minutes" % (str(datetime.datetime.now()), time_passed),
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 
    save_models(generators, discriminators, opt)

    #os.system("test.py --load_from %s --device %s" % (args["save_name"], "cuda:"+str(opt['device'][0])))



