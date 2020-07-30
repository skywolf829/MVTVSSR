from model import *
from options import *
from utility_functions import *
import os
import imageio
import numpy as np
import time
import datetime
from netCDF4 import Dataset

MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

variables = ["cref", "prate","rain", "rain2", "sgs", "sgs2", "sps", "sps2", "srs", "srs2", "sus", "sus2", "svs", "svs2", "sws", "sws2", "uh"]
arr = []
for i in range(1, 202):
    path_to_sim_data = os.path.join(input_folder, "NetCDF", "CM1", "test", "cm1out_00%04d.nc" % (i))
    file2read = Dataset(path_to_sim_data,'r',format="NETCDF4")
    dataframe = file2read.variables[variables[0]][:] 
    for j in range(1, len(variables)):
        dataframe = np.concatenate([dataframe, file2read.variables[variables[j]][:]], axis=0) 
    arr.append(dataframe.astype(np.float32))
    print(dataframe.shape)
    file2read.close()


arr = np.array(arr)
print(arr.shape)
for i in range(arr.shape[0]):
    np.save(os.path.join(input_folder, "CM1", "test", str(i)+".npy"), arr[i].astype(np.float32))
