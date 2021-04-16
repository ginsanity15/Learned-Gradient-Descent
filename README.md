# Learned-Gradient-Descent
This repository contains the PyTorch implementation of the *learned gradient descent (also known as unrolled neural network, UNN)* reported in the article [Solving ill-posed inverse problems using iterative deep neural networks](https://arxiv.org/abs/1704.04058). 

Specifically, the inverse problem being investigated is the reconstruction of the X-ray transmission computed tomography (X-ray CT). As Pytorch does not contain the forward operator of X-ray CT, the forward model needs to be implemented by ourselves. Although writing a customized operator in Pytorch is the most effective approach to do so, such process can be time-consuming. As a proof of concept, I have opted for directly computing the forward projection matrix and loaded it into Python as a giant matrix. 

Nevertheless, it is worth mentioning that such approach is only feasible when we are working on a 2D reconstruction plane with relatively few pixels. In order to perform high resolution reconstruction in the 3D space, it is necessary to write the customized layer.

## Objective
Implement an interesting algorithm for fun :grimacing:

## Contents
This repository contains the following contents:

### Codes
`ForwardMatrix.py`: Generation of the forward matrix

`DataGen.py`: Generation of the training data

`UNN.py`: Training of the learned gradient descent

### Results
Comparisons between the Gradient Descent reconstructions and Learned Gradient Descent reconstructions of random ellipses and Shepp Logan Phantoms

Trained UNN model

## Dependencies
The code is implemented in PyTorch 1.8.0 and the training data is generated with the same method as described in [Solving ill-posed inverse problems using iterative deep neural networks](https://arxiv.org/abs/1704.04058), which requires a third party library called [ODL](https://odlgroup.github.io/odl/). 

As ODL is not mandatory for the implementation of the algorithm, users could also download the data used for the training of the UNN upload in this repository from the following [link](), which contains 1000 phantoms made of random ellipses and their corresponding sinograms generated with `DataGen.py` in this repository. 

## Contact
Bo Gao, PhD student

Ghent University

bo.gao@ugent.be
