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
from matplotlib.pyplot import cm
import time


def bilinear_interpolate(im, x, y):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2]-1)
    x1 = torch.clamp(x1, 0, im.shape[2]-1)
    y0 = torch.clamp(y0, 0, im.shape[3]-1)
    y1 = torch.clamp(y1, 0, im.shape[3]-1)
    
    Ia = im[0, :, x0, y0 ]
    Ib = im[0, :, x1, y0 ]
    Ic = im[0, :, x0, y1 ]
    Id = im[0, :, x1, y1 ]
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
    return Ia*wa + Ib*wb + Ic*wc + Id*wd

def lagrangian_transport(VF, x_res, y_res, time_length, ts_per_sec):
    #x = torch.arange(-1, 1, int(VF.shape[2] / x_res), dtype=torch.float32).unsqueeze(1).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)

    x = torch.arange(0, VF.shape[2], int(VF.shape[2] / x_res), dtype=torch.float32).view(1, -1).repeat([x_res, 1])
    x = x.view(1,x_res, y_res)
    #y = torch.arange(-1, 1, int(VF.shape[3] / y_res), dtype=torch.float32).unsqueeze(0).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)
    y = torch.arange(0, VF.shape[3], int(VF.shape[3] / y_res), dtype=torch.float32).view(-1, 1).repeat([1, y_res])
    y = y.view(1, x_res,y_res)
    particles = torch.cat([x, y],axis=0)
    particles = torch.reshape(particles, [2, -1]).transpose(0,1)
    particles = particles.to("cuda")
    #print(particles)
    particles_over_time = []
    
    
    for i in range(0, time_length * ts_per_sec):
        particles_over_time.append(particles.clone())
        start_t = time.time()
        flow = bilinear_interpolate(VF, particles[:,0], particles[:,1])
        particles[:] += flow[0:2, :].permute(1,0) * (1 / ts_per_sec)
        particles[:] += torch.tensor(list(VF.shape[2:])).to("cuda")
        particles[:] %= torch.tensor(list(VF.shape[2:])).to("cuda")
    particles_over_time.append(particles)
    
    return particles_over_time

def viz_pathlines(frame, pathlines, name, color):
    arr = np.zeros(frame.shape)
    arrs = []

    for i in range(len(pathlines)):
        arrs.append(arr.copy()[0].swapaxes(0,2).swapaxes(0,1))
        for j in range(pathlines[i].shape[0]):
            arr[0, :, int(pathlines[i][j, 0]), int(pathlines[i][j, 1])] = color
    arrs.append(arr.copy()[0].swapaxes(0,2).swapaxes(0,1))
    imageio.mimwrite(name + ".gif", arrs)
    return arrs

def pathline_distance(pl1, pl2):
    d = 0
    for i in range(len(pl1)):
        d += torch.norm(pl1[i] - pl2[i], dim=1).sum()
    return d

def toImg(vectorField, renorm_channels = False):
    vf = vectorField.copy()
    if(len(vf.shape) == 3):
        if(vf.shape[0] == 1):
            return cm.coolwarm(vf[0]).swapaxes(0,2).swapaxes(1,2)
        elif(vf.shape[0] == 2):
            vf += 1
            vf *= 0.5
            vf = vf.clip(0, 1)
            z = np.zeros([1, vf.shape[1], vf.shape[2]])
            vf = np.concatenate([vf, z])
            return vf
        elif(vf.shape[0] == 3):
            if(renorm_channels):
                for j in range(vf.shape[0]):
                    vf[j] -= vf[j].min()
                    vf[j] *= (1 / vf[j].max())
            return vf
    elif(len(vf.shape) == 4):
        return toImg(vf[:,:,:,0], renorm_channels)

def to_mag(vectorField, normalize=True, max_mag = None):
    vf = vectorField.copy()
    r = np.zeros(vf.shape)
    for i in range(vf.shape[0]):
        r[0] += vf[i]**2
    r[0] **= 0.5
    if(normalize):
        if max_mag is None:
            r[0] *= (2 / r[0].max())
        else:
            r[0] *= (2 / max_mag)
    return r[0:1]

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
    sqrtmse = ((GT - fake) ** 2).mean() ** 0.5
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