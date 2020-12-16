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
import gc

def current_mem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):                
                print(obj.name, obj.size(), obj.element_size() * obj.nelement())
            elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.size(), obj.element_size() * obj.nelement())
        except:
            pass

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

def trilinear_interpolate(im, x, y, z, periodic=False):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    z0 = torch.floor(z).type(dtype_long)
    z1 = z0 + 1
    
    if(periodic):
        x1_diff = x1-x
        x0_diff = 1-x1_diff  
        y1_diff = y1-y
        y0_diff = 1-y1_diff
        z1_diff = z1-z
        z0_diff = 1-z1_diff

        x0 %= im.shape[2]
        y0 %= im.shape[3]
        z0 %= im.shape[4]

        x1 %= im.shape[2]
        y1 %= im.shape[3]
        z1 %= im.shape[4]
        
    else:
        x0 = torch.clamp(x0, 0, im.shape[2]-1)
        x1 = torch.clamp(x1, 0, im.shape[2]-1)
        y0 = torch.clamp(y0, 0, im.shape[3]-1)
        y1 = torch.clamp(y1, 0, im.shape[3]-1)
        z0 = torch.clamp(z0, 0, im.shape[4]-1)
        z1 = torch.clamp(z1, 0, im.shape[4]-1)
        x1_diff = x1-x
        x0_diff = x-x0    
        y1_diff = y1-y
        y0_diff = y-y0
        z1_diff = z1-z
        z0_diff = z-z0
    
    c00 = im[0,:,x0,y0,z0] * x1_diff + im[0,:,x1,y0,z0]*x0_diff
    c01 = im[0,:,x0,y0,z1] * x1_diff + im[0,:,x1,y0,z1]*x0_diff
    c10 = im[0,:,x0,y1,z0] * x1_diff + im[0,:,x1,y1,z0]*x0_diff
    c11 = im[0,:,x0,y1,z1] * x1_diff + im[0,:,x1,y1,z1]*x0_diff

    c0 = c00 * y1_diff + c10 * y0_diff
    c1 = c01 * y1_diff + c11 * y0_diff

    c = c0 * z1_diff + c1 * z0_diff
    return c
    
    
def lagrangian_transport(VF, x_res, y_res, time_length, ts_per_sec, device):
    #x = torch.arange(-1, 1, int(VF.shape[2] / x_res), dtype=torch.float32).unsqueeze(1).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)

    x = torch.arange(0, VF.shape[2], int(VF.shape[2] / x_res), dtype=torch.float32).view(1, -1).repeat([x_res, 1])
    x = x.view(1,x_res, y_res)
    #y = torch.arange(-1, 1, int(VF.shape[3] / y_res), dtype=torch.float32).unsqueeze(0).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)
    y = torch.arange(0, VF.shape[3], int(VF.shape[3] / y_res), dtype=torch.float32).view(-1, 1).repeat([1, y_res])
    y = y.view(1, x_res,y_res)
    particles = torch.cat([x, y],axis=0)
    particles = torch.reshape(particles, [2, -1]).transpose(0,1)
    particles = particles.to(device)
    #print(particles)
    particles_over_time = []
    
    
    for i in range(0, time_length * ts_per_sec):
        particles_over_time.append(particles.clone())
        start_t = time.time()
        flow = bilinear_interpolate(VF, particles[:,0], particles[:,1])
        particles[:] += flow[0:2, :].permute(1,0) * (1 / ts_per_sec)
        particles[:] += torch.tensor(list(VF.shape[2:])).to(device)
        particles[:] %= torch.tensor(list(VF.shape[2:])).to(device)
    particles_over_time.append(particles)
    
    return particles_over_time

def lagrangian_transport3D(VF, x_res, y_res, z_res, 
time_length, ts_per_sec, device, periodic=False):
    #x = torch.arange(-1, 1, int(VF.shape[2] / x_res), dtype=torch.float32).unsqueeze(1).expand([int(VF.shape[2] / x_res), int(VF.shape[3] / y_res)]).unsqueeze(0)

    x = torch.arange(0, VF.shape[2], VF.shape[2] / x_res, 
    dtype=torch.float32).view(-1, 1, 1).repeat([1, y_res, z_res])
    x = x.view(1,x_res,y_res, z_res)
    y = torch.arange(0, VF.shape[3], VF.shape[3] / y_res, 
    dtype=torch.float32).view(1, -1, 1).repeat([x_res, 1, z_res])
    y = y.view(1,x_res,y_res, z_res)
    z = torch.arange(0, VF.shape[4], VF.shape[4] / z_res, 
    dtype=torch.float32).view(1, 1, -1).repeat([x_res, y_res, 1])
    z = z.view(1,x_res,y_res, z_res)

    particles = torch.cat([x, y, z],axis=0)
    particles = torch.reshape(particles, [3, -1]).transpose(0,1)
    particles = particles.to(device)
    particles_over_time = []
        
    for i in range(0, time_length * ts_per_sec):
        particles_over_time.append(particles.clone())
        start_t = time.time()
        flow = trilinear_interpolate(VF, particles[:,0], particles[:,1], particles[:,2])
        particles[:] += flow[:, :].permute(1,0) * (1 / ts_per_sec)
        if(periodic):
            particles[:] += torch.tensor(list(VF.shape[2:])).to(device)
            particles[:] %= torch.tensor(list(VF.shape[2:])).to(device)
        else:
            particles[:] = torch.clamp(particles, 0, VF.shape[2])
    particles_over_time.append(particles)
    
    return particles_over_time

def viz_streamlines(frame, streamlines, name, color):
    arr = np.zeros(frame.shape)
    arrs = []

    for i in range(len(streamlines)):
        arrs.append(arr.copy()[0].swapaxes(0,2).swapaxes(0,1))
        for j in range(streamlines[i].shape[0]):
            arr[0, :, int(streamlines[i][j, 0]), int(streamlines[i][j, 1])] = color
    arrs.append(arr.copy()[0].swapaxes(0,2).swapaxes(0,1))
    imageio.mimwrite(name + ".gif", arrs)
    return arrs

def streamline_distance(pl1, pl2):
    d = torch.norm(pl1[0] - pl2[0], dim=1).sum()
    for i in range(1, len(pl1)):
        d += torch.norm(pl1[i] - pl2[i], dim=1).sum()
    return d

def streamline_err_volume(real_VF, rec_VF, res, ts_per_sec, time_length, device, periodic=False):
    
    x = torch.arange(0, real_VF.shape[2], 1, 
    dtype=torch.float32).view(-1, 1, 1).repeat([1, real_VF.shape[3], real_VF.shape[4]])
    x = x.view(real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 1)

    y = torch.arange(0, real_VF.shape[3], 1, 
    dtype=torch.float32).view(1, -1, 1).repeat([real_VF.shape[2], 1, real_VF.shape[4]])
    y = y.view(real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 1)

    z = torch.arange(0, real_VF.shape[4], 1, 
    dtype=torch.float32).view(1, 1, -1).repeat([real_VF.shape[2], real_VF.shape[3], 1])
    z = z.view(real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 1)

    particles_real = torch.cat([x, y, z],axis=3).to(device)
    particles_rec = particles_real.clone()
    
    transport_loss_volume = torch.zeros([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4]], device=device)
    
    
    for i in range(0, time_length * ts_per_sec):

        if(periodic):
            flow_real = trilinear_interpolate(real_VF, 
            particles_real.reshape([-1, 3])[:,0] % real_VF.shape[2], 
            particles_real.reshape([-1, 3])[:,1] % real_VF.shape[3], 
            particles_real.reshape([-1, 3])[:,2] % real_VF.shape[4], periodic = periodic)
            
            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec.reshape([-1, 3])[:,0] % rec_VF.shape[2], 
            particles_rec.reshape([-1, 3])[:,1] % rec_VF.shape[3], 
            particles_rec.reshape([-1, 3])[:,2] % rec_VF.shape[4], periodic = periodic)

            particles_real += flow_real.transpose(0,1).reshape([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 3])
            particles_rec += flow_rec.transpose(0,1).reshape([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4], 3])
            transport_loss_volume += torch.norm(particles_real-particles_rec, dim=3)
        else:
            indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
            (particles_real[:,2] > 0.0) & (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
            (particles_rec[:,2] > 0.0) & (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
            (particles_real[:,2] < real_VF.shape[4]) & (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) & \
            (particles_rec[:,2] < rec_VF.shape[4] ) 
            
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[indices,0], particles_real[indices,1], particles_real[indices,2], 
            periodic = periodic)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[indices,0], particles_rec[indices,1], particles_rec[indices,2], 
            periodic = periodic)

            particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
            
            transport_loss_volume += torch.norm(particles_real[indices] -particles_rec[indices], dim=1).transpose(0, 1).reshape([real_VF.shape[2], real_VF.shape[3], real_VF.shape[4]])
    
    #print("t_init: %0.07f, t_interp: %0.05f, t_add: %0.07f, t_total: %0.07f" % (t_create_particles, t_interp, t_add, time.time()-t_start))
    return transport_loss_volume / (time_length * ts_per_sec)

def streamline_loss3D(real_VF, rec_VF, x_res, y_res, z_res, ts_per_sec, time_length, device, periodic=False):
    
    t_start = time.time()
    t = time.time()
    particles_real = torch.rand([3,x_res*y_res*z_res]).to(device).transpose(0,1)
    particles_real[:,0] *= real_VF.shape[2]
    particles_real[:,1] *= real_VF.shape[3]
    particles_real[:,2] *= real_VF.shape[4]
    particles_rec = particles_real.clone()
    t_create_particles = time.time() - t
    t_add = 0
    t_interp = 0
    transport_loss = torch.autograd.Variable(torch.tensor(0.0).to(device))

    for i in range(0, time_length * ts_per_sec):

        if(periodic):
            t = time.time()
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[:,0] % real_VF.shape[2], 
            particles_real[:,1] % real_VF.shape[3], 
            particles_real[:,2] % real_VF.shape[4], periodic = periodic)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[:,0] % rec_VF.shape[2], 
            particles_rec[:,1] % rec_VF.shape[3], 
            particles_rec[:,2] % rec_VF.shape[4], periodic = periodic)
            t_interp += time.time() - t

            t = time.time()
            particles_real += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec += flow_rec.permute(1,0) * (1 / ts_per_sec)

            transport_loss += torch.norm(particles_real -particles_rec, dim=1).mean()
            t_add += time.time() - t
        else:
            indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
            (particles_real[:,2] > 0.0) & (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
            (particles_rec[:,2] > 0.0) & (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
            (particles_real[:,2] < real_VF.shape[4]) & (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) & \
            (particles_rec[:,2] < rec_VF.shape[4] ) 
            
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[indices,0], particles_real[indices,1], particles_real[indices,2], 
            periodic = periodic)

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[indices,0], particles_rec[indices,1], particles_rec[indices,2], 
            periodic = periodic)

            particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
            
            transport_loss += torch.norm(particles_real[indices] -particles_rec[indices], dim=1).mean()
    
    #print("t_init: %0.07f, t_interp: %0.05f, t_add: %0.07f, t_total: %0.07f" % (t_create_particles, t_interp, t_add, time.time()-t_start))
    return transport_loss / (time_length * ts_per_sec)

def adaptive_streamline_loss3D(real_VF, rec_VF, error_volume, n, octtree_levels,
ts_per_sec, time_length, device, periodic=False):
    
    e_total = error_volume.sum()
    particles_real = torch.zeros([3, n], device=device)
    current_spot = 0
    octtreescale = 3
    #for octtreescale in range(octtree_levels):
    domain_size_x = int((1.0 / (2**octtreescale)) * error_volume.shape[0])
    domain_size_y = int((1.0 / (2**octtreescale)) * error_volume.shape[1])
    domain_size_z = int((1.0 / (2**octtreescale)) * error_volume.shape[2])
    
    for x_start in range(0, error_volume.shape[0], domain_size_x):
        for y_start in range(0, error_volume.shape[1], domain_size_y):
            for z_start in range(0, error_volume.shape[2], domain_size_z):
                error_in_domain = error_volume[x_start:x_start+domain_size_x,
                y_start:y_start+domain_size_y,z_start:z_start+domain_size_z].sum() / e_total
                n_particles_in_domain = int(n * error_in_domain)
                
                particles_real[:,current_spot:current_spot+n_particles_in_domain] = \
                torch.rand([3,n_particles_in_domain])

                particles_real[0,current_spot:current_spot+n_particles_in_domain] *= \
                domain_size_x
                particles_real[1,current_spot:current_spot+n_particles_in_domain] *= \
                domain_size_y
                particles_real[2,current_spot:current_spot+n_particles_in_domain] *= \
                domain_size_z

                particles_real[0,current_spot:current_spot+n_particles_in_domain] += \
                x_start
                particles_real[1,current_spot:current_spot+n_particles_in_domain] += \
                y_start
                particles_real[2,current_spot:current_spot+n_particles_in_domain] += \
                z_start
                current_spot += n_particles_in_domain
                '''
                for i in range(n_particles_in_domain):
                    particles_real[:,current_spot] = torch.rand([3]) * domain_size
                    particles_real[0,current_spot] += x_start
                    particles_real[1,current_spot] += y_start
                    particles_real[2,current_spot] += z_start
                    current_spot += 1
                '''
    particles_real[:,current_spot:] = torch.rand([3, particles_real.shape[1]-current_spot])
    particles_real[0,current_spot:] *= error_volume.shape[0]
    particles_real[1,current_spot:] *= error_volume.shape[1]
    particles_real[2,current_spot:] *= error_volume.shape[2]
        
    particles_real = particles_real.transpose(0,1)
    particles_rec = particles_real.clone()
    
    transport_loss = torch.autograd.Variable(torch.tensor(0.0).to(device))
    for i in range(0, time_length * ts_per_sec):

        if(periodic):
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[:,0] % real_VF.shape[2], 
            particles_real[:,1] % real_VF.shape[3], 
            particles_real[:,2] % real_VF.shape[4])

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[:,0] % rec_VF.shape[2], 
            particles_rec[:,1] % rec_VF.shape[3], 
            particles_rec[:,2] % rec_VF.shape[4])

            particles_real += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec += flow_rec.permute(1,0) * (1 / ts_per_sec)

            transport_loss += torch.norm(particles_real-particles_rec, dim=1).mean()
        else:
            indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
            (particles_real[:,2] > 0.0) & (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
            (particles_rec[:,2] > 0.0) & (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
            (particles_real[:,2] < real_VF.shape[4]) & (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) & \
            (particles_rec[:,2] < rec_VF.shape[4] ) 
            
            flow_real = trilinear_interpolate(real_VF, 
            particles_real[indices,0], particles_real[indices,1], particles_real[indices,2])

            flow_rec = trilinear_interpolate(rec_VF, 
            particles_rec[indices,0], particles_rec[indices,1], particles_rec[indices,2])

            particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
            particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
            
            transport_loss += torch.norm(particles_real[indices]-particles_rec[indices], dim=1).mean()
    return transport_loss / (time_length * ts_per_sec)
    
def sample_adaptive_streamline_seeds(error_volume, n, device):
    e_total = error_volume.sum()
    particles = torch.zeros([3, n], device=device)
    current_spot = 0
    octtreescale = 3
    #for octtreescale in range(octtree_levels):
    domain_size_x = int((1.0 / (2**octtreescale)) * error_volume.shape[0])
    domain_size_y = int((1.0 / (2**octtreescale)) * error_volume.shape[1])
    domain_size_z = int((1.0 / (2**octtreescale)) * error_volume.shape[2])
    
    for x_start in range(0, error_volume.shape[0], domain_size_x):
        for y_start in range(0, error_volume.shape[1], domain_size_y):
            for z_start in range(0, error_volume.shape[2], domain_size_z):
                error_in_domain = error_volume[x_start:x_start+domain_size_x,
                y_start:y_start+domain_size_y,z_start:z_start+domain_size_z].sum() / e_total
                n_particles_in_domain = int(n * error_in_domain)
                
                
                particles[:,current_spot:current_spot+n_particles_in_domain] = \
                torch.rand([3,n_particles_in_domain])
                
                particles[0,current_spot:current_spot+n_particles_in_domain] *= domain_size_x
                particles[1,current_spot:current_spot+n_particles_in_domain] *= domain_size_y
                particles[2,current_spot:current_spot+n_particles_in_domain] *= domain_size_z

                particles[0,current_spot:current_spot+n_particles_in_domain] += x_start
                particles[1,current_spot:current_spot+n_particles_in_domain] += y_start
                particles[2,current_spot:current_spot+n_particles_in_domain] += z_start
                current_spot += n_particles_in_domain

    particles[:,current_spot:] = torch.rand([3, particles.shape[1]-current_spot])
    particles[0,current_spot:] *= error_volume.shape[0]
    particles[1,current_spot:] *= error_volume.shape[1]
    particles[2,current_spot:] *= error_volume.shape[2]

    particles = particles.type(torch.LongTensor).transpose(0,1)
    particles[:,0] = torch.clamp(particles[:,0], 0, error_volume.shape[0]-1)
    particles[:,1] = torch.clamp(particles[:,1], 0, error_volume.shape[1]-1)
    particles[:,2] = torch.clamp(particles[:,2], 0, error_volume.shape[2]-1)

    particle_volume = torch.zeros(error_volume.shape).type(torch.FloatTensor).to(device)
    for i in range(particles.shape[0]):
        particle_volume[particles[i,0],particles[i,1],particles[i,2]] += 1
    
    return particle_volume

def streamline_loss2D(real_VF, rec_VF, x_res, y_res, ts_per_sec, time_length, device, periodic=False):
    x = torch.arange(0, real_VF.shape[2], real_VF.shape[2] / x_res, 
    dtype=torch.float32).view(-1, 1).repeat([1, y_res])
    x = x.view(1,x_res,y_res)
    y = torch.arange(0, real_VF.shape[3], real_VF.shape[3] / y_res, 
    dtype=torch.float32).view(1, -1).repeat([x_res, 1])
    y = y.view(1,x_res,y_res)
    
    particles_real = torch.cat([x, y],axis=0)
    particles_real = torch.reshape(particles_real, [2, -1]).transpose(0,1)
    particles_real = particles_real.to(device)

    particles_rec = torch.cat([x, y],axis=0)
    particles_rec = torch.reshape(particles_rec, [2, -1]).transpose(0,1)
    particles_rec = particles_rec.to(device)
    
    transport_loss = torch.autograd.Variable(torch.tensor(0.0).to(device))
    for i in range(0, time_length * ts_per_sec):
        indices = (particles_real[:,0] > 0.0) & (particles_real[:,1] > 0.0) & \
        (particles_rec[:,0] > 0.0) & (particles_rec[:,1] > 0.0) & \
        (particles_real[:,0] < real_VF.shape[2]) & (particles_real[:,1] < real_VF.shape[3]) & \
        (particles_rec[:,0] < rec_VF.shape[2]) & (particles_rec[:,1] < rec_VF.shape[3]) 
        
        flow_real = bilinear_interpolate(real_VF, 
        particles_real[indices,0], particles_real[indices,1])

        flow_rec = bilinear_interpolate(rec_VF, 
        particles_rec[indices,0], particles_rec[indices,1])

        particles_real[indices] += flow_real.permute(1,0) * (1 / ts_per_sec)
        particles_rec[indices] += flow_rec.permute(1,0) * (1 / ts_per_sec)
        print(indices.sum())
        if(periodic):
            particles_real[:] += torch.tensor(list(real_VF.shape[2:])).to(device)
            particles_real[:] %= torch.tensor(list(real_VF.shape[2:])).to(device)
            particles_rec[:] += torch.tensor(list(rec_VF.shape[2:])).to(device)
            particles_rec[:] %= torch.tensor(list(rec_VF.shape[2:])).to(device)
        else:
            #with torch.no_grad():
            #    particles_real = torch.clamp(particles_real, 0, real_VF.shape[2])
            #    particles_rec = torch.clamp(particles_rec, 0, rec_VF.shape[2])
            transport_loss += torch.norm(particles_real[indices] -particles_rec[indices], dim=1).mean()
    return transport_loss / (time_length * ts_per_sec)

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

def to_vort(vectorField, normalize=True, device=None):
    vf = vectorField.clone()
    if(len(vf.shape) == 4):
        xdy = spatial_derivative2D(vf[:,0:1,:,:], 0, device)
        ydx = spatial_derivative2D(vf[:,1:2,:,:], 1, device)
        return (ydx - xdy)[0,0]
    elif(len(vf.shape) == 5):
        
        zdy = spatial_derivative3D(vf[:,2:3,:,:,:], 1, device)
        ydz = spatial_derivative3D(vf[:,1:2,:,:,:], 0, device)

        
        xdz = spatial_derivative3D(vf[:,0:1,:,:,:], 0, device)
        zdx = spatial_derivative3D(vf[:,2:3,:,:,:], 2, device)

        xdy = spatial_derivative3D(vf[:,0:1,:,:,:], 1, device)
        ydx = spatial_derivative3D(vf[:,1:2,:,:,:], 2, device)

        vorts = torch.cat([zdy-ydz, xdz-zdx, ydx-xdy], axis=0)
        return vorts

def feature_distance(img1, img2, device):
    if(features_model is None):
        model = models.vgg19(pretrained=True).to(device=device)
        model.eval()
        layer = model.features

    img1 = np.expand_dims(img1.swapaxes(0,2).swapaxes(1,2), axis=0)
    img2 = np.expand_dims(img2.swapaxes(0,2).swapaxes(1,2), axis=0)

    if(img1.shape[1] == 1):
        img1 = np.repeat(img1, 3, axis=1)    
    if(img2.shape[1] == 1):
        img2 = np.repeat(img2, 3, axis=1)

    img1_tensor = np2torch(img1, device=device)
    img1_feature_vector = layer(img1_tensor).cpu().detach().numpy()

    img2_tensor = np2torch(img2, device=device)
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
def compute_gradients(input_frame, device):
    # X gradient filter
    x_gradient_filter = torch.Tensor([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]).to(device)

    x_gradient_filter = x_gradient_filter.view((1,1,3,3))
    G_x = F.conv2d(input_frame, x_gradient_filter, padding=1)

    y_gradient_filter = torch.Tensor([[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]).to(device)

    y_gradient_filter = y_gradient_filter.view((1,1,3,3))
    G_y = F.conv2d(input_frame, y_gradient_filter, padding=1)

    return G_x, G_y

def compute_laplacian(input_frame):
    # X gradient filter
    laplacian_filter = torch.Tensor([[0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]]).to(device)

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