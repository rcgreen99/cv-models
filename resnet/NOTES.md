# ResNet

ResNet, from the paper Deep Residual Learning for Image Recognition, was a major advancement in deep learning architectures. It was the first model to have a depth over 100 and still perform well. In fact, its depth was what enabled it to become the new SOTA for CNNs.

## Abstract

- Deeper neural networks are more difficult to train.
- Authors present residual learning to make training deeper models easier
- New SOTA on ImageNet. 8x deeper than VGG

## Introduction

- Deeper networks perform better in general
- Major blocker to increasing depth is the vanishing/exploding gradient problem
    - This problem was largely solved by normalized initialization and intermediate normalization layers
- Adding more layers past a point actually decreases accuracy -- even on the the _training set_. 
    - The authors claim that this doesn't make sense, as the models should be at least able to achieve on-par performance.
- To get around this issue, they propose deep residual learning (i.e. residual connections)

## Related Work

- Residual vectors have been used in image recognition via VLAD
- Other neural networks have used shortcut connections.
    - One model used a connection between input and the last hidden layer
    - Another type of neural network, "highway networks," present shorcut functions with gating functions. 
    However, these gates are data dependent and have parameters, while ResNet's residual connections are identity mappings and are parameter free.

## Deep Residual Learning

### Residual Learning

- Consider $H(x)$ to be the mapping fit by a few stacked layers where $x$ is the inputs to the first of these layers (i.e. $H(x)$ is the manifold that fits the data)
- Since we can hypothesize that multiple nonlinear layers can asymoptotically approximate compliated functions, then it is equalitvalent to hypothesize that the same is true for $H(x) - x.$
- So, we can expect these stacked layers to be able to approximate the residual function $F(x) := H(x) - x$
- This means $H(x) = F(x) + x$
- The authors are careful to point out that while the hypothesis's suggest that both should asymptotically approximate the desired functions, it might be easier to train one or the other.

- The authors revisit the idea that adding identity mapping layers to the end of a network should allow it to perform exactly the same as before.
However, this is found not to be true. Which suggests that the models have difficulty approximating the identity mapping.

### Identity Mapping by Shorcuts

- Residual learning is added to every few stacked layers, forming a building block. Formally, $$y = F(x, {W_i}) + x$$
- Where $x$ and $y$ are the inpout and output vectors of the layers
- The function $F(x, {W_i})$ represents the residual mapping to be learned.
- Importantly, the shorcut connections add neither extra parameter or computational complexity (the element-wise addition is negligible)
- The dimensions of $x$ and $F$ must be equal of course. So, if $F$ is bigger or smaller than $x$ (e.g. when changing the input/output channels) it's dimensions must be altered
$$ y = F(x, {W_i}) + W_sx$$

### Network Architectures

- To test the performance of residual nets, they compared "plain" networks of the same configuration as the residual networks just without the residual connections.
- Inspired by VGG, the conv layers mostly use 3x3 filters and follow two simple design rules
    - for the same output feature map size, the layers have the same number of filters
    - if the feature map size is halved, the number of filters is doubled soas to preserve the time complexity per layer
- Downsampling is performed by conv layers that have a stride of 2
- For residual connections where the dimensions do not increase, the identity shorcuts can be directly used (simply added to the output)
    - When the dimensions do increase, there are two options.
        1. Perform identity mapping with padding zeros (adds no extra parameter)
        2. Projection shorcut to match dimensinos using 1x1 conv layer with stride 2
    - The authors experimented with using just zero padding, identity and projectino, and just projections and found that identity and projection was better than zero padding, and only projections were slightly better than the combination of identity and projection. The authors suggest this is due to the increased weight count, and therefore choose the combined identity and projections for the rest of the paper.
- The network ends with a 1000-way fully-connected layer with softmax
- The model has fewer filters than VGG-19 despite having 34 layers (for this comparison in particular) and uses only 18% the FLOPs

#### Implementation

- Image is resized with shorter side randomly sampled in [256, 480] for scale agumentation
- A 224x224 crop is randomly sampled from an image or its hoirzontal flip with per pixel mean subtracted
- Batch normalization after each conv layer and before ReLU
- SGD with mini-btach size of 256
- Learning rate starts from 0.1 and is divided by 10 when the error plateaus
- Weight decay of 0.0001 and a momentum of 0.9
- Do not use dropout
