# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:23:30 2021

@author: gaobo

In this script, we compute the forward matrix A 

"""

import numpy as np
import matplotlib.pyplot as plt
        
#%%
num_angles = 120

sample = 100
sample_length = 10.0
sample_size = sample_length/sample
sam_x = np.linspace(-sample/2, sample/2, sample+1)*sample_size
sam_y = np.linspace(-sample/2, sample/2, sample+1)*sample_size

num_det = 300
det_length = 30.0
det_size = det_length/det_length
det_min = -(num_det/2-1)*det_size - det_size/2.0
det_max = -det_min
det_pos = np.linspace(det_min, det_max, num_det)*det_size

sod = 100.0
odd = 100.0

angles = list(np.linspace(0, 2*np.pi-2*np.pi/num_angles, num_angles))
A = np.zeros([num_angles*num_det, sample*sample])

#%%
# TODO: implement the distance-driven forward projection
# Compute the forward matrix with Brute force
for a, angle in enumerate(angles):
    # Determine Source and Detector positions
    source_pos = np.array([np.cos(angle)*sod, np.sin(angle)*sod])
    det_cen = np.array([-np.cos(angle)*odd, -np.sin(angle)*odd])
    det_pos_x = np.cos(angle - np.pi/2)*det_pos + det_cen[0]
    det_pos_y = np.sin(angle - np.pi/2)*det_pos + det_cen[1]
    det_pos_xy = np.array([det_pos_x, det_pos_y])
    
    # Determine the direction of source toward each detectorlet
    source_dir = np.array([det_pos_x-source_pos[0], det_pos_y-source_pos[1]])
    for s in range(source_dir.shape[1]):
        source_vec = source_dir[:,s]
        nor_source = source_vec/np.linalg.norm(source_vec)
        for i in range(sample*sample):
            x_i = i%sample
            y_i = int(i/sample)
            temp_sample_x = np.array(sam_x[x_i:x_i+2])
            temp_sample_y = np.array(sam_y[y_i:y_i+2])
            
            sample_x_dis = (temp_sample_x - source_pos[0])/nor_source[0]
            sample_y_dis = (temp_sample_y - source_pos[1])/nor_source[1]
            
            inter_min = max(min(sample_x_dis), min(sample_y_dis))
            inter_max = min(max(sample_x_dis), max(sample_y_dis))
            
            if(inter_max > inter_min):
                index1 = a*num_det + s
                A[index1, i] = inter_max - inter_min
    
    print(str(a+1) + " angle completed")

#%%
# Sanity check through compute the projection of a square
sample = np.zeros([sample, sample])
upper = int(0.3*sample)
lower = int(0.7*sample)
sample[upper:lower, upper:lower] = 10

sino = np.dot(A, sample.ravel())
plt.imshow(sino.reshape([num_angles, num_det]))

#%%
file_forward = r'D:\DL\UnrolledNeuralNetwork\ForwardMatrix.npy'
np.save(file_forward, A.astype(np.float32))