# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:33:13 2021

@author: bobgao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from libtiff import TIFF

#%%
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
class Net(nn.Module):
    def __init__(self, memory):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(memory+3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, memory+1, 3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        s = F.relu(x[:, 0:-1])
        return x[:, -1], s

class UnrolledNeuralNetwork(nn.Module):
    def __init__(self, A, iteration, memory = 5, device = torch.device('cpu')):
        super(UnrolledNeuralNetwork, self).__init__()
        self.Nets = nn.ModuleList()
        self.iteration = iteration
        self.A = A.to(device)
        self.num_memory = memory
        self.device = device
        
        for i in range(iteration):
            tempNet = Net(self.num_memory).to(self.device)
            self.Nets.append(tempNet)
            
    def forward(self, y): # y: proj_data
        b = y.size()[0]
        m = self.num_memory
        f = torch.zeros([b, 10000]).to(self.device)
        div = torch.zeros([b, 1, 100, 100]).to(self.device)
        self.memories = torch.zeros([b, m, 100, 100]).to(self.device)
        for i in range(self.iteration):
            corr = torch.matmul(f, self.A.T) - y
            back_proj = torch.matmul(corr, self.A)
            recons = f.reshape([b, 1, 100, 100])
            div = self.Divergence(recons)
            inputs = torch.zeros([b, m+3, 100, 100])
            inputs[:,0:1] = recons
            inputs[:,1:m+1] = self.memories
            inputs[:,m+1:m+2] = back_proj.reshape([b, 1, 100, 100])
            inputs[:,m+2:m+3] = div
            tempNet = self.Nets[i]
            delta_f, self.memories = tempNet(inputs.to(self.device))
            f += delta_f.reshape([b, 10000]) 
            
        return f
    
    def Divergence(self, image):  
        assert len(image.size()) == 4, "Shape of image must be: [batch_size, 1, height, width]"
        
        b, m, w, h = image.size()
        
        def gradx(image):
            res = torch.zeros([b, m, w, h]).to(self.device)
            res[:,:,1:] = image[:,:,1:] - image[:,:,0:-1]
            return res
        
        def grady(image):
            res = torch.zeros([b, m, w, h]).to(self.device)
            res[:,:,:,1:] = image[:,:,:,1:] - image[:,:,:,0:-1]
            return res
        
        grad_x = gradx(image)
        grad_y = grady(image)
        
        output = torch.zeros([b, m, w, h]).to(self.device)
        output[:,:,1:] = -1*(grad_x[:,:,1:] - grad_x[:,:,0:-1])
        output[:,:,:,1:] += -1*(grad_y[:,:,:,1:] - grad_y[:,:,:,0:-1])
        return output
    
#%%
# Load in forward matrix A
filename = r'D:\DL\UnrolledNeuralNetwork\BrutalForce\ForwardMatrix.npy'
a = np.load(filename)
A = torch.from_numpy(a)

#%%
class DatasetCT(Dataset):
    def __init__(self, root_dir, transform = None):
        """
        Parameters
        ----------
        root_dir : Directory to all the images (features and labels)
        transform :Optional transform to beh applied on a sample
        Returns
        -------
        """
        self.root_dir = root_dir
        self.feature_files = os.listdir(root_dir + '\\Sino')
        self.label_files = os.listdir(root_dir + '\\Phan')
        self.transform = transform
        
    def __len__(self):
        return len(self.feature_files)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir + '\\Sino' + '\\' + self.feature_files[idx])
        phan_name = os.path.join(self.root_dir + '\\Phan' + '\\' + self.label_files[idx])
        sino = ImageReader(img_name)
        phan = ImageReader(phan_name)
        sample = {'sino': sino, 'phan': phan}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample    

# Data transform operator
class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """
    def __call__(self, sample):
        sino, phan = sample['sino'], sample['phan']
        
        return {'sino': torch.from_numpy(sino),
                'phan': torch.from_numpy(phan)}

class ToVector(object):
    """" Convert ndarrays to vectors """
    def __call__(self, sample):
        sino, phan = sample['sino'], sample['phan']
        return {'sino': torch.reshape(sino, [36000]), 
                'phan': torch.reshape(phan, [10000])}
        
#%%
root_dir = r'D:\DL\UnrolledNeuralNetwork\BrutalForce\ToySet'
train_data = DatasetCT(root_dir, transform= transforms.Compose([ToTensor(), 
                                                                ToVector()]))

fig = plt.figure()
for i in range(4):
    sample = train_data[i]
    print(i, sample['sino'].size(), sample['phan'].size())
    
    ax = plt.subplot(2, 2, i+1)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['phan'].reshape([100, 100]))
#%%
# Load in training/validation/testing dataset
batch_size = 16
data_loader = torch.utils.data.DataLoader(train_data, 
                                          batch_size = batch_size, shuffle=True)

#for i_batch, sample_batched in enumerate(data_loader):
#    print(i_batch, sample_batched['sino'].size(), sample_batched['gt'].size())
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

iteration = 10 # Iteration of the learned gradient descent
memory = 5
UNN = UnrolledNeuralNetwork(A, iteration, memory, device)
UNN.to(device)

UNN_total_params = [p.numel() for p in UNN.parameters()]

criterion = nn.MSELoss()
#optimizer = optim.Adam(UNN.parameters(),lr=0.00001)
#optimizer = optim.SGD(UNN.parameters(),lr=0.001,momentum=0.9)

train_losses = []
val_losses = []
#%%
# Train the network
batches = len(data_loader)
train = int(0.8*batches)
val = batches - train

for epoch in range(200):
    running_loss_train = 0.0
    running_loss_val = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data['sino'].to(device), data['phan'].to(device)
        
        if(epoch < 100):
            optimizer = optim.Adam(UNN.parameters(), lr=2e-5)
        else:
            optimizer = optim.Adam(UNN.parameters(), lr=1e-5)
            
        if(i < train):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = UNN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss_train += loss.item()
        else:
            outputs = UNN(inputs)
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()
            
    print('Epoch: %d, train loss: %.3f, val loss: %.3f' %
          (epoch+1, running_loss_train, running_loss_val))
    train_losses.append(running_loss_train)
    val_losses.append(running_loss_val)
    running_loss_train = 0.0
    running_loss_val = 0.0

print('Finished Training')

#%%
path = r'D:\DL\UnrolledNeuralNetwork\Github\Results\UNN.pt'
torch.save(UNN.state_dict(), path)

#%%
val_losses_weighted = []
for loss in val_losses:
    val_losses_weighted.append(loss*4)

plt.figure()
plt.title("Losses in the training and validation set")
plt.semilogy(train_losses[5::])
plt.semilogy(val_losses_weighted[5::])

#%%
def TensorToNumpy(tensor):
    return tensor.cpu().detach().numpy()

#%%
""" Code below shows the reconstruction outcomes of UNN on validation set"""

num_sino = outputs.size()[0]
# sinos = inputs.reshape([num_sino, 120, 300]).cpu().detach().numpy()
recons = outputs.reshape([num_sino, 100, 100]).cpu().detach().numpy()
gt = labels.reshape([num_sino, 100, 100]).cpu().detach().numpy()

# bp = torch.matmul(inputs, A.to(device)).reshape([num_sino, 100, 100]).cpu().detach().numpy()
# est_proj = torch.matmul(labels, A.T.to(device)).reshape([num_sino, 120, 300]).cpu().detach().numpy()

idx = np.random.randint(0, num_sino-1)
plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.title('Learned Gradient Descent')
plt.imshow(recons[idx], vmin = 0, vmax = gt[idx].max(), cmap='gray'), plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('Grount Truth')
plt.imshow(gt[idx], vmin = 0, cmap='gray'), plt.colorbar()

#%%

"""Code below compares the performance of UNN and Gradient Descent on Shepp Logan Phantom"""


#%%
def GradientDescentTorch(sino, a, iterations, alpha = 0.002):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    sino_t = torch.from_numpy(sino).to(device)
    a_t = torch.from_numpy(a).to(device)
    recon = torch.zeros([10000]).to(device)
    
    for i in range(iterations):
        corr = torch.matmul(a_t, recon) - sino_t
        recon -= alpha*torch.matmul(a_t.T, corr)
        if i%10 == 9:
            print(str(i) + " iteration finished")
    
    return recon.reshape([100, 100]).cpu().detach().numpy()

#%%
temp_sino = r'D:\DL\UnrolledNeuralNetwork\BrutalForce\Results\SheppLogan\TestSet\SheppLoganSino.tif'
sino = ImageReader(temp_sino)
sino_t = torch.from_numpy(sino).reshape([1, 36000])

recon_gd = GradientDescentTorch(sino.reshape([36000]), a, 500)

recon = UNN(sino_t.to(device))
recon_unn = recon.reshape([100, 100]).cpu().detach().numpy()

#%%
plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.title('Learned Gradient Descent')
plt.imshow(np.rot90(recon_unn), vmin = 0, vmax = 1.0, cmap='gray'), plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('Gradient Descent 500 iterations')
plt.imshow(np.rot90(recon_gd), vmin = 0, vmax = 1.0, cmap='gray'), plt.colorbar()

#%%
from PIL import Image
filename = r'D:\DL\UnrolledNeuralNetwork\Github\Results\SheppLogan\Recons\UNN.tif'
#filename = r'D:\DL\UnrolledNeuralNetwork\BrutalForce\Results\SheppLogan\Recons\GD_10iters.tif'
im = Image.fromarray(np.fliplr(np.rot90(recon_np)))
im.save(filename)