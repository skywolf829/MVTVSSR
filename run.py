from model import *
from options import *
from utility_functions import *
import os
import imageio
import numpy as np
import time
import datetime
from netCDF4 import Dataset

import matplotlib.pyplot as plt

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

'''
variables = ["cref", 
"prate",
"rain", 
#"rain2", 
"sgs", 
#"sgs2", 
"sps", 
#"sps2", 
"srs", 
#"srs2", 
"sus", 
#"sus2", 
"svs", 
#"svs2", 
"sws", 
#"sws2", 
"uh"]

arr = []
for i in range(2, 402):
    path_to_sim_data = os.path.join(input_folder, "NetCDF", "CM1", "validation", "cm1out_00%04d.nc" % (i))
    file2read = Dataset(path_to_sim_data,'r',format="NETCDF4")
    dataframe = file2read.variables[variables[0]][:] 
    for j in range(1, len(variables)):
        dataframe = np.concatenate([dataframe, file2read.variables[variables[j]][:]], axis=0) 
    dataframe = np.array(dataframe)
    np.save(os.path.join(input_folder, "CM1_2", "validation", str(i-2)+".npy"), dataframe.astype(np.float32))
    file2read.close()

'''


variables = ["dbz", 
"ncg",
"nci",  
"ncr", 
"ncs", 
"prs", 
"qc", 
"qg", 
"qi", 
"qr", 
"qs", 
"qv", 
"th", 
"uinterp",
"vinterp",
"winterp",
"xvort",
"yvort",
"zvort"]

arr = []
for i in range(2, 402):
    path_to_sim_data = os.path.join(input_folder, "NetCDF", "CM1", "test", "cm1out_00%04d.nc" % (i))
    file2read = Dataset(path_to_sim_data,'r',format="NETCDF4")
    dataframe = file2read.variables[variables[0]][:] 
    for j in range(1, len(variables)):
        thisvar = file2read.variables[variables[j]][:]        
        dataframe = np.concatenate([dataframe, thisvar], axis=0) 
    dataframe = np.array(dataframe)
    np.save(os.path.join(input_folder, "CM1_3D", "test", str(i-2)+".npy"), dataframe.astype(np.float32))
    file2read.close()



'''
variables = [
    "QCLOUD",
    "QGRAUP",
    "QICE",
    "QSNOW",
    "QVAPOR",
    "CLOUD",
    "PRECIP",
    "P",
    "TC",
    "U",
    "V",
    "W",
]
for i in range(1, 49):
    ts = np.zeros([13, 100, 500, 500])
    print("Timestep %i" % i)
    for j in range(len(variables)):
        fname = "E:\Isabel\%sf%02d.bin" % (variables[j], i)
        f=open(fname, "rb")
        data = np.fromfile(f, '>f4')
        
        data = data.reshape([100, 500, 500]).swapaxes(1,2).astype(np.float32)
        ts[j] = data

    np.save("E:\\Isabel\\" + str(i-1)+'.npy', ts)
'''

'''
for i in range(0, 48):
    data = np.load("E:\\Isabel\\" + str(i)+'.npy')
    traindata = data[:,50,:,:]
    validationdata = data[:,45,:,:]
    testdata = data[:,55,:,:]
    np.save(os.path.join(input_folder, "Isabel", "train", "%i.npy"%i), traindata)
    np.save(os.path.join(input_folder, "Isabel", "validation", "%i.npy"%i), validationdata)
    np.save(os.path.join(input_folder, "Isabel", "test", "%i.npy"%i), testdata)
'''