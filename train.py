from model import *
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

    parser.add_argument('--mode',default=None,help='The type of input - 2D, 3D, or 2D time-varying')
    parser.add_argument('--data_folder',default=None,type=str,help='File to train on')
    parser.add_argument('--validation_folder',default=None,type=str,help='File to validate on')
    parser.add_argument('--num_training_examples',default=None,type=int,help='Frames to use from training file')
    parser.add_argument('--save_folder',default=None, help='The folder to save the models folder into')
    parser.add_argument('--save_name',default=None, help='The name for the folder to save the model')
    parser.add_argument('--num_channels',default=None,type=int,help='Number of channels to use')
    parser.add_argument('--downscale_ratio',default=None,type=float,help='Ratio for x downscaling')
    parser.add_argument('--min_dimension_size',default=None,type=int,help='Minimum dimension size')

    parser.add_argument('--num_blocks',default=None,type=int, help='Num of conv-batchnorm-relu blocks per gen/discrim')
    parser.add_argument('--base_num_kernels',default=None,type=int, help='Num conv kernels in lowest layer')
    parser.add_argument('--conv_layer_padding',default=None, type=int,help='Padding for each conv layer')
    parser.add_argument('--network_input_padding',default=None,type=int, help='Padding before entering network')
    parser.add_argument('--kernel_size',default=None, type=int,help='Conv kernel size')
    parser.add_argument('--stride',default=None, type=int,help='Conv stride length')
    parser.add_argument('--padding_method',default=None, type=str,help='layer, zero, pbc, or raw')

    
    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--device',nargs="+",type=int,default=None, help='Device to use')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--ranking',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--save_generators',default=None, type=str2bool,help='Whether or not to save generators')
    parser.add_argument('--save_discriminators',default=None, type=str2bool,help='Whether or not to save discriminators')
    parser.add_argument('--save_conditioner',default=None, type=str2bool,help='Whether or not to save the conditioner')

    parser.add_argument('--alpha',default=None, type=float,help='Reconstruction loss coefficient')
    parser.add_argument('--generator_steps',default=None, type=int,help='Number of generator steps to take')
    parser.add_argument('--discriminator_steps',default=None, type=int,help='Number of discriminator steps to take')
    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--minibatch',default=None, type=int,help='Size of minibatch to train on')
    parser.add_argument('--num_workers',default=None, type=int,help='Number of workers for dataset loader')
    parser.add_argument('--learning_rate',default=None, type=float,help='Learning rate for the networks')
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')
    parser.add_argument('--gamma',default=None, type=float,help='')
    parser.add_argument('--lambda_grad',default=None, type=float,help='')

    parser.add_argument('--use_fourier_scales',default=None, type=str2bool,help='Whether or not to use scales found from Fourier analysis')
    parser.add_argument('--num_fourier_k_means',default=None,type=int, help='Number of scales to use for fourier analysis')
    parser.add_argument('--fourier_top_percent',default=None, type=float,help='Use the top percent of coefficients as input to the k means')

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

    if(opt["train_distributed"]):
        # Set up distributed training variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12359'

    # Determine scales
    init_scales(opt)

    # Init models
    generators = []
    discriminators_s = []
    discriminators_t = []
    
    now = datetime.datetime.now()
    start_time = time.time()
    print_to_log_and_console("Started training at " + str(now), 
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")
    opt["num_training_examples"] = len(os.listdir(os.path.join(input_folder,opt["data_folder"])))

    # Train each scale 1 by 1
    for i in range(opt["n"]-1):

        start_time_scale_n = time.time()

        print_to_log_and_console(str(datetime.datetime.now()) + " - Beginning training on scale " + str(len(generators)),
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt")

        if(opt["train_distributed"]):
            mp.spawn(
                train_single_scale, nprocs=len(opt["device"]), 
                args=(generators, discriminators_s, discriminators_t, opt)
            )
            opt = load_options(os.path.join(save_folder, opt["save_name"]))
            generators, discriminator_s, discriminator_t = load_models(opt, "cpu")
        else:
            generator, discriminator_s, discriminator_t = train_single_scale(0, generators, discriminators_s, discriminators_t, opt)
            generators.append(generator)
            discriminators_s.append(discriminator_s)
            discriminators_t.append(discriminator_t)
            
        time_passed = (time.time() - start_time_scale_n) / 60
        print_to_log_and_console("%s - Finished training in scale %i in %f minutes" % (str(datetime.datetime.now()), len(generators)-1, time_passed),
        os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 


    time_passed = (time.time() - start_time) / 60
    print_to_log_and_console("%s - Finished training  in %f minutes" % (str(datetime.datetime.now()), time_passed),
    os.path.join(opt["save_folder"], opt["save_name"]), "log.txt") 
    save_models(generators, discriminators_s, discriminators_t, opt)

    #os.system("test.py --load_from %s --device %s" % (args["save_name"], "cuda:"+str(opt['device'][0])))



