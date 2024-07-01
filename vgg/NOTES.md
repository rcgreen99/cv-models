# VGG

In 2014, paper Very Deep Convolutional Networks for Large-Scale Image Recognition was released by Karen Simonyan and Andrew Zisserma of the University of Oxford. The model they outline in the paper, "VGG" (for Visual Geometry Group a group within the Department of Engineering at Oxford), is only 16-19 layers in depth.
This "very deep" CNN is considered tiny by today's standards, but at the time it achieved a new best on the ImageNet Challenge in 2014. 

## Abstract

The primary contribution of the paper is the author's invesitgation on the effect of depth with CNNs. They used "very small" (3x3) convolutional filters (kernels), which allowed them to push the depth from the 5-8 layers of AlexNet, to 16-19.

## Introduction

GPUs and large public image repositories led the increasing success of CNNs from 2012-2014 (and of course for years to come). ImageNet in particular was very importtant for comparing models.

## ConvNet Configurations

### Architecture
- Input is 224x224 RGB image.
- Only preprocessing is subtracting the mean RGB value (computed on training set) from each pixel
- Uses 3x3 kernels, 1 pixel stride, 1 pixel padding
- Spatial pooling is carried out by five-max pooling layers, which follow _some_ of the conv. layers
- Max-pooling is perforemd over a 2x2 pixel window, with stride 2
- Conv layers aer followed by Fully-Connected (FC) layers. First two have 4096 channels each, the third performs 1000 way ILSVRC classification
- The final layer is a soft-max layer

- All hidden layers are equipped with retification (ReLU) nonlinearity. 
- None of the networks use Local Resposne Normalisation (LRN)
- The width of conv layers is small, starting from 64 in the first layer and then increasing by a factor 2 after each maxa-pooling layer, until it reaches 512

### Discussion on kernel size

Previous models used rather large 11x11 or 7x7 kernels. VGG uses only 3x3.

>  It is easy to see that a stack of two 3x3 conv. layers (without spatial pooling in between) has an effective receptive field of 5x5; three such layers have a 7x7 effective receptive field. So what have we gained by using, for instance, a stack of three 3x3 conv. layers instead of a single 7x7 layer? First, we incorporate three ReLU's instead of one, which makes the decision functon more discriminative. Second, we decrease the nubmerof parameters: assuming that both the input and the output of a three-layer 3x3 convolutional stack has channels, the stack is parametrise d dby $3(3^2C^2)=26C^2$ weights; a single 7x7 conv. layer would require $7^2C^2=49C^2$ parameters, i.e. 81% more. This can be seen as imposing a regularisation on the 7x7 conv. filters, forcing them to hae a decomposition through the 3x3 filters (with non-linearity injected in between).

### 1x1 conv. layers

1x1 conv. layers (used in one of the 16 layer versions) are a way to increase the nonlinearity of the decision function without affecting the receptive fields of the conv. layers.

## Classification Framework

### Training
