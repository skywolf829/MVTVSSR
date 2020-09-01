from model import *
from options import *
from utility_functions import *
import torch.nn.functional as F
import torch
from piq import ssim, psnr, multi_scale_ssim
import os
import imageio
import argparse
from typing import Union, Tuple
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from math import log, e



MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(MVTVSSR_folder_path, "InputData", "Synthetic", "train")
test_folder = os.path.join(MVTVSSR_folder_path, "InputData", "Synthetic", "test")


dX, dY = 128, 128
xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))

def randColor():
    return np.array([random.random(), random.random(), random.random()]).reshape((1, 1, 3))
def getX(): return xArray
def getY(): return yArray
def safeDivide(a, b):
    return np.divide(a, np.maximum(b, 0.001))

functions = [(0, randColor),
             (0, getX),
             (0, getY),
             (1, np.sin),
             (1, np.cos),
             (2, np.add),
             (2, np.subtract),
             (2, np.multiply),
             (2, safeDivide)]
depthMin = 5
depthMax = 8

def buildImg(depth = 0):
    funcs = [f for f in functions if
                (f[0] > 0 and depth < depthMax) or
                (f[0] == 0 and depth >= depthMin)]
    nArgs, func = random.choice(funcs)
    args = [buildImg(depth + 1) for n in range(nArgs)]
    return func(*args)

k = 0
while k < 500:
    img = buildImg()
    img -= img.min()
    if(img.max() != 0):
        img *= (1/img.max())
        img *= 2
        img -= 1
        s = (int(dX / img.shape[0]), int(dY / img.shape[1]), int(3 / img.shape[2]))
        img = np.tile(img, s)
        
        name = os.path.join(train_folder, str(k)+".npy")
        np.save(name, img.swapaxes(0,2).swapaxes(1,2))
        k += 1

k = 0
while k < 100:
    img = buildImg()
    img -= img.min()
    if(img.max() != 0):
        img *= (1/img.max())
        img *= 2
        img -= 1
        s = (int(dX / img.shape[0]), int(dY / img.shape[1]), int(3 / img.shape[2]))
        img = np.tile(img, s)
        
        name = os.path.join(test_folder, str(k)+".npy")
        np.save(name, img.swapaxes(0,2).swapaxes(1,2))
        k += 1




