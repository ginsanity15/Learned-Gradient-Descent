# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:37:44 2021

@author: bobgao
"""

import numpy as np
import odl
import os
from PIL import Image
import torch
from libtiff import TIFF
#%%
def random_ellipse():
    return ((np.random.rand() - 0.3) * np.random.exponential(0.3),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            np.random.rand() - 0.5, np.random.rand() - 0.5,
            np.random.rand() * 2 * np.pi)

def random_phantom(spc):
    n = np.random.poisson(100)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.geometric.ellipsoid_phantom(spc, ellipses)

def ImageReader(filename, shape = None):
    tif   = TIFF.open(filename, mode = 'r')
    temp = tif.read_image()
    descr = tif.GetField('ImageDescription')
    if(descr is not None):
        s = descr.split()
        slope     = np.float32(s[2])
        offset    = np.float32(s[5])
        temp = temp*slope+offset
    tif.close()
    if shape is not None:
        temp = temp.reshape(shape)
    return temp
#%%
""" Generation of phantoms made of random ellipses """
gt_path = r'D:\DL\UnrolledNeuralNetwork\Github\Test\Data'

space = odl.uniform_discr([-5, -5], [5, 5], [100, 100], dtype='float32')
for i in range(10000):
    phan = random_phantom(space)
    phan_file = gt_path + '\\phan_' + str(i).zfill(4) + '.tif'
    im_p = Image.fromarray(phan.asarray())
    im_p.save(phan_file)

#%%
# Torch can be used as GPU version of numpy, load forward matrix A to GPU
filename = r'D:\DL\UnrolledNeuralNetwork\BrutalForce\ForwardMatrix.npy'
a = np.load(filename)
A = torch.from_numpy(a)

#%%
""" Generation of the sinograms corresponding to the random ellipses """
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

sino_path = r'D:\DL\UnrolledNeuralNetwork\ToySet\Sinos'
gt_path = r'D:\DL\UnrolledNeuralNetwork\ToySet\Phans'

phan_files = os.listdir(gt_path)
phantoms = torch.zeros([len(phan_files), 10000]).to(device)

for i, phan in enumerate(phan_files):
    phan_file = gt_path + '\\' + phan
    temp_phan = ImageReader(phan_file)
    phantoms[i, :] = torch.from_numpy(temp_phan).reshape([10000])
    
sinos_t = torch.matmul(phantoms, A.T.to(device))
sinos = sinos_t.cpu().detach().numpy()
for i in range(sinos.shape[0]):
    sino_file = sino_path + '\\sino_' + str(i).zfill(4) + '.tif' 
    temp_sino = sinos[i].reshape([120, 300])
    sino_g = np.random.normal(temp_sino, 0.05)
    im = Image.fromarray(sino_g)
    im.save(sino_file)
    