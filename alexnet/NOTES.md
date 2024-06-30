# AlexNet

AlexNet is famous for starting the neural network research frenzy of the last decade and a half. Released in 2012, it achieved SOTA on the ImageNet image classification competition--the first time a neural network would do so. In the subsequent years, we would see lots of improvements to CNN's, with ImageNet becoming dominated by neural networks.

Notably, Ilya Sutskever and Geoffrey E. Hinton are listed as authors after Alex Krizhevsky, whom AlexNet is named after.

## Abstract

- Trained a 5 layer convolutional neural network on ImageNet LSVRC-2010
- Achieved SOTA for top-1 error rate of 37.5%, and a top-5 error rate of 17%.
- CNN was 5 layers, 60 million parameters, and 650,000 neurons
- Introduced "dropout" to reduce overfitting


## Introduction

- To learn about thousands of objects from millions of images, we need a model with a alrg elearning capacity.
- CNNs are good for this as their capacity can be controlled by varying their depth and breath, and have already been proved to work on images for some tasks
- Additionally, they have many fewer connections and parameters than similarly sized Feed Foward neural networks
- AlexNet proved to be a new SOTA for ImageNet. In fact, even with the 1.2million training examples, it would overfit, so multiple methods to control this weere used.
- The main limitation of the network size ws the amount of memory available on current GPUs

## The Dataset

- ImageNet consists of 15 million high-resolution images belonging to 22,000 categores
- ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) stgarted in 2010
- ILSVRC uses a subset of ImageNet
- It consists of roughly 1000 images per 1000 categories, totallying righly 1.2 million training images, 50,000 validation images, and 150,000 test images.
- 

## The Architecture 

- 8 layers: 1st 5 are CNN and last 3 are fully connected
- ReLU nonlinearity applied after each layer
- Trained on multiple GPUs (2 GTX 580s)
- Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the sam kernel map
- AlexNet employs overlapping pooling, which they found reduces overfitting slightly

### Details

- 1st CNN layer filters the 224x224x3 input image with 96 kernels of size 11x11x3 wit hstride of 4 pixels
- 2nd CNN takes as input the (response-normalized and pooled) output from the first CNN layer and filters it with 256 kernels of size 5x5x48
- 3rd, 4th, and 5th layers are connected without any pooling or normalization layers
- 3rd layer has 384 kernels of size 3x3x256 connected to the (normalized and pooled) oupts of the 2nd layer
- 4th layer has 384 kernels of 3x3x192
- 5th has 256 kernels of 3x3x192
- fully connected layers have 4096 neurons each
- finaly layer is 1000-way softmax
