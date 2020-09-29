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


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

a = np.load(os.path.join(input_folder, "JHUturbulence", "isotropic1024coarse", "0.npy"))
np.save(os.path.join(input_folder, "JHUturbulence", "isotropic128coarse", "0.npy"), a[:,::8,::8])
'''
print(a.shape)
print(a[0].max())
print(a[0].min())
print(a[1].max())
print(a[1].min())
print(a[2].max())
print(a[2].min())


plt.hist(a[0].flatten(), 50, histtype='step', stacked=True, fill=False, color='blue')
plt.hist(a[1].flatten(), 50, histtype='step', stacked=True, fill=False, color='green')
plt.hist(a[2].flatten(), 50, histtype='step', stacked=True, fill=False, color='orange')
plt.legend(['u', 'v', 'w'])
plt.title('data distribution - no scaling')
plt.show()

mags = np.linalg.norm(a, axis=0)
m_mag = mags.max()

b = (a / m_mag)
plt.hist(b[0].flatten(), 50, histtype='step', stacked=True, fill=False, color='blue')
plt.hist(b[1].flatten(), 50, histtype='step', stacked=True, fill=False, color='green')
plt.hist(b[2].flatten(), 50, histtype='step', stacked=True, fill=False, color='orange')
plt.legend(['u', 'v', 'w'])
plt.title('data distribution - magnitude scaled')
plt.show()
#imageio.imwrite("b", ((b+1) / 2).swapaxes(0,2).swapaxes(0,1))

c = a.copy()
c[0] -= c[0].min()
c[0] /= c[0].max()
c[0] -= 0.5
c[0] *= 2
c[1] -= c[1].min()
c[1] /= c[1].max()
c[1] -= 0.5
c[1] *= 2
c[2] -= c[2].min()
c[2] /= c[2].max()
c[2] -= 0.5
c[2] *= 2

#imageio.imwrite("c", ((c+1) / 2).swapaxes(0,2).swapaxes(0,1))
plt.hist(c[0].flatten(), 50, histtype='step', stacked=True, fill=False, color='blue')
plt.hist(c[1].flatten(), 50, histtype='step', stacked=True, fill=False, color='green')
plt.hist(c[2].flatten(), 50, histtype='step', stacked=True, fill=False, color='orange')
plt.legend(['u', 'v', 'w'])
plt.title('data distribution - min/max')
plt.show()

d = b.copy()
d[0] -= d[0].mean()
d[1] -= d[1].mean()
d[2] -= d[2].mean()
#imageio.imwrite("d", ((d+1) / 2).swapaxes(0,2).swapaxes(0,1))
plt.hist(d[0].flatten(), 50, histtype='step', stacked=True, fill=False, color='blue')
plt.hist(d[1].flatten(), 50, histtype='step', stacked=True, fill=False, color='green')
plt.hist(d[2].flatten(), 50, histtype='step', stacked=True, fill=False, color='orange')
plt.legend(['u', 'v', 'w'])
plt.title('data distribution - magnitude scaled with mean shift')
plt.show()
'''