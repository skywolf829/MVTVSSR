import imageio
import os
import numpy as np
import torch
import torch.nn as nn
from skimage.measure import compare_ssim
from math import log10
import scipy
import math
import numbers
from torch.nn import functional as F

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


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

# https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/4
def compute_gradients(input_frame):
    # X gradient filter
    x_gradient_filter = torch.Tensor([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]).cuda()

    x_gradient_filter = x_gradient_filter.view((1,1,3,3))
    G_x = F.conv2d(input_frame, x_gradient_filter, padding=1)

    y_gradient_filter = torch.Tensor([[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]).cuda()

    y_gradient_filter = y_gradient_filter.view((1,1,3,3))
    G_y = F.conv2d(input_frame, y_gradient_filter, padding=1)

    return G_x, G_y

def compute_laplacian(input_frame):
    # X gradient filter
    laplacian_filter = torch.Tensor([[0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]]).cuda()

    laplacian_filter = laplacian_filter.view((1,1,3,3))
    laplacian = F.conv2d(input_frame, laplacian_filter, padding=1)

    return laplacian

def create_graph(x, y, title, xlabel, ylabel, colors, labels):    
    import matplotlib.pyplot as plt
    handles = []
    for i in range(len(y)):
        h, = plt.plot(x, y[i], c=colors[i])
        handles.append(h)
    plt.legend(handles, labels, loc='upper right', borderaxespad=0.)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()