import os
import json

class Options():
    def get_default():
        opt = {}
        # Input info
        opt["mode"]                    = "3D"      # What SinGAN to use - 2D or 3D
        opt["data_folder"]             = "JHUturbulence/isotropic128_downsampled"
        #opt["data_folder"]             = "TestImage"
        opt["image_normalize"]         = False
        opt["scale_data"]              = False
        opt['scale_on_magnitude']      = True
        opt["save_folder"]             = "SavedModels"
        opt["save_name"]               = "Temp"    # Folder that the model will be saved to
        opt["num_channels"]            = 3
        opt["spatial_downscale_ratio"] = 0.5       # Spatial downscale ratio between levels
        opt["min_dimension_size"]      = 32        # Smallest a dimension can go as the smallest level
        opt["train_date_time"]         = None      # The day/time the model was trained (finish time)
        
        # GAN info
        opt["num_blocks"]              = 5
        opt["base_num_kernels"]        = 32        # Num of kernels in smallest scale conv layers
        opt["pre_padding"]             = False         # Padding on conv layers in the GAN
        opt["kernel_size"]             = 3
        opt["stride"]                  = 1
        opt['conv_groups']             = 1
        opt['separate_chans']          = False
        
        opt["n"]                       = 0         # Number of scales in the heirarchy, defined by the input and min_dimension_size
        opt["resolutions"]             = []        # The scales for the GAN
        opt["noise_amplitudes"]        = []
        opt["downsample_mode"]         = "nearest"
        opt["upsample_mode"]           = "trilinear"

        opt["train_distributed"]       = False
        opt["device"]                  = "cuda:0"
        opt["gpus_per_node"]           = 1
        opt["num_nodes"]               = 1
        opt["ranking"]                 = 0
        opt["save_generators"]         = True
        opt["save_discriminators"]     = True
        opt["physical_constraints"]    = "none"
        opt["patch_size"]              = 128
        opt["training_patch_size"]     = 128
        opt["regularization"]          = "GP" #Either TV (total variation) or GP (gradient penalty) or SN 
        # GAN training info
        opt["alpha_1"]                 = 0       # Reconstruction loss coefficient
        opt["alpha_2"]                 = 0        # Adversarial loss coefficient
        opt["alpha_3"]                 = 0        # Soft physical loss coefficient
        opt["alpha_4"]                 = 1        # mag_and_angle loss
        opt["alpha_5"]                 = 1          # first derivative loss coeff
        opt["alpha_6"]                 = 0  # Lagrangian transport loss
        opt["adaptive_streamlines"]    = False
        opt['streamline_res']          = 100
        opt['streamline_length']       = 50
        opt['periodic']                = True
        opt["generator_steps"]         = 3
        opt["discriminator_steps"]     = 3
        opt["epochs"]                  = 2000
        opt["minibatch"]               = 1        # Minibatch for training
        opt["num_workers"]             = 0
        opt["learning_rate"]           = 0.0005    # Learning rate for GAN
        opt["beta_1"]                  = 0.5
        opt["beta_2"]                  = 0.999
        opt["gamma"]                   = 0.1
        opt['zero_noise']              = True

        # Info during training (to continue if it stopped)
        opt["scale_in_training"]       = 0
        opt["iteration_number"]        = 0
        opt["save_every"]              = 100
        opt["save_training_loss"]      = True

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    

def load_options(load_location):
    opt = Options.get_default()
    print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt
