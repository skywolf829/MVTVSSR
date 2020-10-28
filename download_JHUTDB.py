from model import *
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
from skimage.transform.pyramids import pyramid_reduce
from skimage.feature import match_template
import cv2 as cv


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

import zeep
import struct
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')
token="edu.osu.buckeyemail.wurster.18-92fb557b" #replace with your own token

def get_frame(x_start, x_end, y_start, y_end, z_start, z_end, 
sim_name, timestep, field, num_components):
    #print(x_start)
    #print(x_end)
    #print(y_start)
    #print(y_end)
    #print(z_start)
    #print(z_end)
    result=client.service.GetAnyCutoutWeb(token,sim_name, field, timestep,
                                            x_start+1, y_start+1, z_start+1, x_end, y_end, z_end,
                                            1, 1, 1, 0, "")  # put empty string for the last parameter
    # transfer base64 format to numpy
    nx=x_end-x_start
    ny=y_end-y_start
    nz=z_end-z_start
    base64_len=int(nx*ny*nz*num_components)
    base64_format='<'+str(base64_len)+'f'

    result=struct.unpack(base64_format, result)
    result=np.array(result).reshape((nz, ny, nx, num_components))
    return result, x_start, x_end, y_start, y_end, z_start, z_end

'''

import pyJHTDB
from pyJHTDB import libJHTDB

lJHTDB = libJHTDB()
lJHTDB.initialize()
lJHTDB.add_token(token)

def get_frame_big(xstart, ystart, zstart, 
xend, yend, zend, xstep, ystep, zstep,
sim_name, 
timestepstart, timestepend, timestepstep, 
field, num_components):
    r = lJHTDB.getbigCutout(
        data_set=sim_name, fields=field, t_start=timestepstart, 
        t_end=timestepend, t_step=timestepend,
        start=np.array([xstart, ystart, zstart], dtype = np.int),
        end=np.array([xend, yend, zend], dtype = np.int),
        step=np.array([xstep, ystep, zstep], dtype = np.int),
        filter_width=0,
        filename="na")
    return r
'''
def get_full_frame(x_start, x_end, y_start, y_end, z_start, z_end,
sim_name, timestep, field, num_components):
    full = np.zeros((z_end-z_start, y_end-y_start, x_end-x_start, num_components))
    x_len = 1024
    y_len = 1024
    for k in range(z_start, z_end, 1):
        for i in range(x_start, x_end, x_len):
            for j in range(y_start, y_end, y_len):
                x_stop = min(i+x_len, x_end)
                y_stop = min(j+y_len, y_end)
                z_stop = min(k+1, z_end)
                full[k:z_stop,j:y_stop,i:x_stop,:] = get_frame(i,x_stop, j, y_stop, k, z_stop, sim_name, timestep, field, num_components)[0]
    return full

def download_file(url, file_name):
    try:
        html = requests.get(url, stream=True)
        open(f'{file_name}.json', 'wb').write(html.content)
        return html.status_code
    except requests.exceptions.RequestException as e:
       return e

def get_full_frame_parallel(x_start, x_end, y_start, y_end, z_start, z_end,
sim_name, timestep, field, num_components, num_workers):
    threads= []
    full = np.zeros((z_end-z_start, y_end-y_start, x_end-x_start, num_components), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        done = 0
        x_len = 128
        y_len = 128
        z_len = 128
        for k in range(z_start, z_end, z_len):
            for i in range(x_start, x_end, x_len):
                for j in range(y_start, y_end, y_len):
                    x_stop = min(i+x_len, x_end)
                    y_stop = min(j+y_len, y_end)
                    z_stop = min(k+z_len, z_end)
                    threads.append(executor.submit(get_frame, i,x_stop, j, y_stop, k, z_stop, sim_name, timestep, field, num_components))
        for task in as_completed(threads):
           r, x1, x2, y1, y2, z1, z2 = task.result()
           full[z1-z_start:z2-z_start,
           y1-y_start:y2-y_start,
           x1-x_start:x2-x_start,:] = r.astype(np.float32)
           del r
           done += 1
           print("Done: %i/%i" % (done, len(threads)))
    return full

#f = get_full_frame(0, 1024, 0, 1024, 0, 1, "isotropic1024coarse", 1, "u", 3)
#f = get_full_frame_parallel(0, 1024, 0, 1024, 0, 1, "isotropic1024coarse", 1, "u", 3, 16)
#f = get_full_frame_parallel(0, 10240, 0, 1536, 1, 2, "channel5200", 1, "u", 3, 64)

frames = []

#name = "channel"
name = "isotropic1024coarse"
t0 = time.time()
count = 0
endts = 2
ts_skip = 25
for i in range(1, endts, ts_skip):
    print("TS %i/%i" % (i, endts))
    f = get_full_frame_parallel(448, 576, #x
    448, 576, #y
    448, 576, #z
    name, i, 
    "u", 3, 
    64)
    #np.save(os.path.join(input_folder, "JHUturbulence",
    #name,
    #str(count) + ".npy"), f[0].swapaxes(0,2).swapaxes(1,2).astype(np.float32))
    print(f.shape)
    np.save("0.npy", f.astype(np.float32).swapaxes(0,3).swapaxes(3,2).swapaxes(2,1))
    count += 1
    frames.append(f[0])

print("finished")
print(time.time() - t0)
#lJHTDB.finalize()

from netCDF4 import Dataset
rootgrp = Dataset("test.nc", "w", format="NETCDF4")
velocity = rootgrp.createGroup("velocity")
u = rootgrp.createDimension("u")
v = rootgrp.createDimension("v")
w = rootgrp.createDimension("w")
w = rootgrp.createDimension("channels", 3)
us = rootgrp.createVariable("u", f.dtype, ("u","v","w"))
vs = rootgrp.createVariable("v", f.dtype, ("u","v","w"))
ws = rootgrp.createVariable("w", f.dtype, ("u","v","w"))
mags = rootgrp.createVariable("magnitude", f.dtype, ("u","v","w"))
velocities = rootgrp.createVariable("velocities", f.dtype, ("u","v","w", "channels"))
mags[:] = np.linalg.norm(f,axis=3)
#velocities[:] = f


'''
f = np.array(frames)
max_mag = None
mags = np.zeros(f.shape)
for i in range(f.shape[0]):
    for j in range(f.shape[1]):
        for k in range(f.shape[2]):
            mag = (f[i,j,k,0]**2 + f[i,j,k,1]**2 + f[i,j,k,2]**2)**0.5
            mags[i,j,k,0] = mag
            if max_mag is None or mag > max_mag:
                max_mag = mag

mags *= (1 / max_mag)


f[:,:,:,0] -= f[:,:,:,0].min()
f[:,:,:,0] *= (1/ f[:,:,:,0].max())
f[:,:,:,1] -= f[:,:,:,1].min()
f[:,:,:,1] *= (1/ f[:,:,:,1].max())
f[:,:,:,2] -= f[:,:,:,2].min()
f[:,:,:,2] *= (1/ f[:,:,:,2].max())


mags = cm.coolwarm(mags[:,:,:,0])

imageio.mimwrite(name + "_vmag.gif", mags)
imageio.mimwrite(name + ".gif", f)


'''