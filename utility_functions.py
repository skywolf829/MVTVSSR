import imageio
import os
import numpy as np
import torch
from skimage.measure import compare_ssim
from math import log10

def feature_distance(img1, img2):
    if(features_model is None):
        model = models.vgg19(pretrained=True).to(device="cuda")
        model.eval()
        layer = model.features

    img1 = np.expand_dims(img1.swapaxes(0,2).swapaxes(1,2), axis=0)
    img2 = np.expand_dims(img2.swapaxes(0,2).swapaxes(1,2), axis=0)

    if(img1.shape[1] == 1):
        img1 = np.repeat(img1, 3, axis=1)    
    if(img2.shape[1] == 1):
        img2 = np.repeat(img2, 3, axis=1)

    img1_tensor = np2torch(img1, device="cuda")
    img1_feature_vector = layer(img1_tensor).cpu().detach().numpy()

    img2_tensor = np2torch(img2, device="cuda")
    img2_feature_vector = layer(img2_tensor).cpu().detach().numpy()
    
    return np.mean((img1_feature_vector - img2_feature_vector) ** 2)

def MSE(x, y):
    _mse = np.mean((x - y) ** 2)
    return _mse

def PSNR(GT,fake,max_val=255): 
    sqrtmse = np.mean((GT - fake) ** 2) ** 0.5
    return 20 * log10(max_val / sqrtmse)

def SSIM(GT,fake,multichannel=True):
    s = compare_ssim(GT, fake, multichannel=multichannel)
    return s

def RGB2NP(img_array):
    im = img_array.astype(np.float32)
    im *= (2/255)
    im -= 1
    return im

def NP2RGB(np_array):
    im = np_array + 1
    im *= (255/2)
    im = im.astype(np.uint8)
    return im

def np2torch(x, device):    
    x = torch.from_numpy(x)
    x = x.type(torch.FloatTensor)
    return x.to(device)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def print_to_log_and_console(message, location, file_name):
    #print to console
    print(message)
    #create logs directory if needed
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except OSError:
            print ("Creation of the directory %s failed" % location)
    #a for append mode, will also create file if it doesn't exist yet
    _file = open(os.path.join(location, file_name), 'a')
    #print to the file
    print(message, file=_file)

def create_folder(start_path, folder_name):
    f_name = folder_name
    full_path = os.path.join(start_path, f_name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print_to_log_and_console ("Creation of the directory %s failed" % full_path)
    else:
        #print_to_log_and_console("%s already exists, overwriting save " % (f_name))
        full_path = os.path.join(start_path, f_name)
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print_to_log_and_console ("Creation of the directory %s failed" % full_path)
    return f_name