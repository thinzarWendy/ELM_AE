# Extreme Learning Machines (ELMs) for Image Classification and Reconstruction
This repository contains a Jupyter notebook that demonstrates the implementation of Extreme Learning Machines (ELMs) for both image classification and image reconstruction tasks, using Python and PyTorch.

# Overview
Extreme Learning Machines (ELMs) are a type of feedforward neural network that differs from conventional networks in the way they are trained. Unlike traditional neural networks that require iterative backpropagation for training, ELMs have analytically determined output weights, leading to significantly faster training times.

In this notebook, I explore the implementation of ELMs through the following steps:

Image Classification
Random Initialization: Parameters of the hidden nodes (weights w and biases b) are assigned randomly.
Hidden Layer Output: Compute the hidden layer output matrix H.
Output Weights Calculation: Compute the output weights β using the Moore-Penrose pseudoinverse of the hidden layer output matrix.
We first experiment with a hidden layer size of 1000 neurons, and then proceed to test the network with varying numbers of hidden nodes, extending up to 5000 to observe the impact on classification performance.

Image Reconstruction (Autoencoder)
In the second part of the notebook, I modify the ELM implementation to perform image reconstruction. In this task, the network acts as an autoencoder where the input and output are identical, and the hidden layer serves as a bottleneck. This forces the network to learn a compressed, meaningful representation of the input data. The steps followed are similar to those in the image classification task, but with the target output being the same as the input.
