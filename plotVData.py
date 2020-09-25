#%%
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


verts_df = pd.read_csv('InputData/p1r3periodic/vortex.mesh', sep=' ', skiprows=2336, header=None)
data_df  = pd.read_csv('InputData/p1r3periodic/vortex-1-init.gf', sep=' ', skiprows=5, header=None)

verts_a = verts_df.to_numpy()
data_a  = data_df.to_numpy()

print(verts_a.size)
print(data_a.size)

N=int(math.sqrt( data_a.size / 4 ))

x = verts_a[:,0]
y = verts_a[:,1]

z = np.zeros((N,N))
for i in range(0, data_a.size-1):
    xid = (int(round(x[i]*24)) + 24) % 48
    yid = (int(round(y[i]*24)) + 24) % 48

    z[xid,yid] = data_a[i]

print(z)
np.argwhere(np.isnan(z))

X, Y = np.meshgrid(range(0,N), range(0,N))

print(X.shape)
print(Y.shape)
print(z.shape)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z)
plt.show()
