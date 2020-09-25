from SinGAN_models import *
from options import *
from utility_functions import *
import torch.nn.functional as F
import torch
import os
import imageio
import argparse
from typing import Union, Tuple
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from math import log, e
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd

def vf2rgb(frame, scale):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    x = np.meshgrid(np.linspace(-1.0, 1.0, int(frame.shape[1]/scale)))
    y = np.meshgrid(np.linspace(-1.0, 1.0, int(frame.shape[2]/scale)))

    ax.quiver(x, y, frame[0,::scale,::scale], frame[1,::scale,::scale], 
    pivot='middle',linestyle='solid')

    ax.set_title("Vector field")
    ax.axis('off')

    canvas.draw()       # draw the canvas, cache the renderer

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()

    return image.reshape([int(height), int(width), 3])

def data_to_bin_probability(data, num_bins):
    probs = []
    data = data.clone().flatten()
    d_min = -1
    d_max = 1
    for i in range(num_bins):
        bin_min = d_min + i * ((d_max-d_min) / num_bins)
        bin_max = d_min + (i+1) * ((d_max-d_min) / num_bins)
        if(i == num_bins - 1):            
            indices = torch.where((data >= bin_min) & (data <= bin_max))
        else:
            indices = torch.where((data >= bin_min) & (data < bin_max))
        probs.append(indices[0].shape[0] / data.shape[0])
    return probs

def calculate_entropy(probs):
    ent = 0.
    for i in range(len(probs)):
        ent -= probs[i] * log(probs[i] + 1e-8, e)
    return ent


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")


def read_ts(i):
    verts_df = pd.read_csv('InputData/p1r5periodic/vortex.mesh', sep=' ', skiprows=32+shape**2, header=None)
    vx  = pd.read_csv('InputData/p1r5periodic/vortex-2-%i.gf' % i, sep=' ', skiprows=5, header=None)
    vy  = pd.read_csv('InputData/p1r5periodic/vortex-3-%i.gf' % i, sep=' ', skiprows=5, header=None)
    vd  = pd.read_csv('InputData/p1r5periodic/vortex-1-%i.gf' % i, sep=' ', skiprows=5, header=None)

    verts_a = verts_df.to_numpy()
    data_x  = vx.to_numpy()
    data_y  = vy.to_numpy()
    data_d  = vd.to_numpy()

    N=int(math.sqrt( data_x.size / 4 ))

    print(N)
    print(data_x.shape)
    x = verts_a[:,0]
    y = verts_a[:,1]

    dx = np.zeros((1, N,N))
    dy = np.zeros((1, N,N))
    dd = np.zeros((1, N,N))
    for i in range(0, data_x.size-1):
        xid = (int(round(float(x[i])*(N/2))) + int(N/2)) % N
        yid = (int(round(float(y[i])*(N/2))) + int(N/2)) % N

        dx[0, xid,yid] = data_x[i]
        dy[0, xid,yid] = data_y[i]
        dd[0, xid,yid] = data_d[i]

    return dx, dy, dd



max_ts = 2950

shape = 96
vorts = []
mags = []
ds = []
vfs = []

for i in range(1,int(max_ts / 50) + 1):
    print(i*50)
    ts = i * 50
    vx, vy, den = read_ts(ts)

    arr = np.concatenate([vx, vy])
    np.save(os.path.join(input_folder, "first_sim", str(i-1)+".npy"), arr)
    
    vf = vf2rgb(arr, 2)
    vfs.append(vf)
    
    
    mag = (arr[0]*arr[0] + arr[1]+arr[1])**0.5
    mags.append(mag)

    dvydx = np.array(np.gradient(arr[1], axis=0))
    dvxdy = np.array(np.gradient(arr[0], axis=1))
    vort = dvydx - dvxdy
    vorts.append(vort)

    den = np.reshape(den, [shape, shape])
    ds.append(den)

vorts = np.array(vorts)
vorts -= vorts.min()
vorts *= (1/vorts.max())
np.save("vorts.npy", vorts)
vorts = cm.coolwarm(vorts)
imageio.mimwrite("vorticity.gif", vorts.swapaxes(1, 2))

vfs = np.array(vfs)
imageio.mimwrite("vectorfields.gif", vfs)

mags = np.array(mags)
mags -= mags.min()
mags *= (1/mags.max())
np.save("mags.npy", mags)
mags = cm.coolwarm(mags)
imageio.mimwrite("magnitude.gif", mags.swapaxes(1, 2))

ds = np.array(ds)
ds -= ds.min()
ds *= (1/ds.max())
np.save("ds.npy", ds)
ds = cm.coolwarm(ds)
imageio.mimwrite("density.gif", ds.swapaxes(1, 2))

imageio.imwrite("density.png", ds[0].swapaxes(0, 1))


im=imageio.imread("TestImage.jpg")[:,:,0:1].swapaxes(0, 2).swapaxes(1, 2)
np.save("0.npy", im)