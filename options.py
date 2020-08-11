import os
import json

class Options():
    def get_default():
        opt = {}
        # Input info
        opt["mode"]                    = "2D"      # What SinGAN to use - 2D or 3D
        opt["data_folder"]             = "CM1_2/train"
        opt["validation_folder"]       = "CM1_2/validation"
        opt["num_training_examples"]   = None
        opt["save_folder"]             = "SavedModels"
        opt["save_name"]               = "Temp"    # Folder that the model will be saved to
        opt["num_channels"]            = None
        opt["downscale_ratio"]         = 0.5       # Downscale ratio between levels
        opt["min_dimension_size"]      = 32        # Smallest a dimension can go as the smallest level
        opt["train_date_time"]         = None      # The day/time the model was trained (finish time)
        
        # GAN info
        opt["num_blocks"]              = 3
        opt["base_num_kernels"]        = 32        # Num of kernels in smallest scale conv layers
        opt["conv_groups"]             = 1
        opt["conv_layer_padding"]      = 0         # Padding on conv layers in the GAN
        opt["network_input_padding"]   = 5         # Padding to do to input to GAN (ideally half the receptive field)
        opt["kernel_size"]             = 3
        opt["stride"]                  = 1
        opt["padding_method"]          = "zero"    # Either "zero" or "pbc" padding
        opt["noise_amplitudes"]        = []
        
        opt["n"]                       = 5         # Number of scales in the heirarchy, defined by the input and min_dimension_size
        opt["scales"]                  = []        # The scales for the GAN
        opt["base_resolution"]         = [128, 128]# Base resolution for full scale images [rows, cols] aka [y, x] aka [height, width]
        opt["use_spectral_norm"]       = True
        opt["downsample_mode"]         = "nearest"
        opt["upsample_mode"]           = "bilinear"

        opt["train_distributed"]       = False
        opt["device"]                  = [0]
        opt["gpus_per_node"]           = 1
        opt["num_nodes"]               = 1
        opt["ranking"]                 = 0
        opt["save_generators"]         = True
        opt["save_discriminators_s"]   = True
        opt["save_discriminators_t"]   = True

        # GAN training info
        opt["alpha"]                   = 10        # Reconstruction loss coefficient
        opt["generator_steps"]         = 3
        opt["discriminator_steps"]     = 3
        opt["epochs"]                  = 5
        opt["minibatch"]               = 12        # Minibatch for training
        opt["num_workers"]             = 0
        opt["learning_rate"]           = 0.0005    # Learning rate for GAN
        opt["beta_1"]                  = 0.0
        opt["beta_2"]                  = 0.999
        opt["gamma"]                   = 0.1   
        opt["lambda_grad"]             = 0.1       # Coefficient for the gradient loss

        # 2D specific options
        opt["use_fourier_scales"]      = False     # Choose scales of pyramid using Fourier analysis
        opt["num_fourier_k_means"]     = 5         # Num of clusters for k means
        opt["fourier_top_percent"]     = 5         # Use the top % of coefficients as input to the k means

        # Info during training (to continue if it stopped)
        opt["scale_in_training"]       = 0
        opt["iteration_number"]        = 0
        opt["save_every"]              = 500
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
