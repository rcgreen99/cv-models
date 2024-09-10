# Mask R-CNN Whitepaper Notes

## Abstract

Extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. In Janurary 2018, it scored top results in all three of the COCO suite challenges, including instance segmentation, bounding-box detection, and person keypoint detection. Their conceptually simple and minimal overhead to the Faster R-CNN make it a solid basline for instance-level recognition.

## 1. Introduction

Instance segmentation is hard because it combines object detection and semantic segmentation into one task, where each object must be separately segmented.

Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in *parallel* with the existing branch for classification and bounding box regrssion. The mask branch is a small FCN applied to each RoI.

Faster R-CNN was not desinged for pixel-to-pixel alignment (since it is for object detection). This is most evident in how *RoIPool* performs coarse spatial quantization for feature extraction. To fix the misalignment, we propose a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations.

RoIAlign has a large impact, improving mask accuracy by 50%.

## 2. Related Work

### R-CNN

The Region-based CNN (R-CNN, 2014) approach to bounding-box object detection is to attend to a manageable number of candidate object regions, and then run a CNN on each "Region of Interest" (RoI). R-CNN was extended to allow attending to RoIs on feature maps using RoIPool, which made it faster and more accurate (Fast R-CNN). Faster R-CNN improved upon this by learning the attention mechanism with a Region Proposal Network (RPN). As of 2018, Faster R-CNN was leading in multiple benchmarks.

### Instance Segmentation

Multiple previous approaches attempted to perform segmentation first, and then object detection and classification. These approaches are slow and less accurate. Mask-RCNN tries to first detect instances (via RPN) and then segment.

## 3. Mask R-CNN

Faster R-CNN has two outputs for each candidate: class label and bounding box. Mask R-CNN adds a third branch for predicting masks.

### Faster R-CNN

Faster R-CNN has two stages. First is the Region Proposal Network (RPN), which proposes candidate object bounding boxes. The second (and more important) extracts features using RoIPool from each candidate box and performs classification and bounding-box regression. The features used by both stages can be shared for faster inference.

### Mask R-CNN

Mask R-CNN also uses a two stage system, with no modifications to the first stage (RPN). The second stage is modified by adding a third branch *in parallel* to predicting the class and box offset. So, the output bbox and class are in no way derived from the mask prediciton.

#### Loss

During trainin,g the multi-task loss is defined on each sampled RoI as $L = L_{cls} + L_{box} + L_{mask}$. The mask branch has a $Km^2$ dimensional output for each RoI, which encode $K$ binary masks of resolution $m \times m$, onefor each of the $K$ classes. That means that the more $K$ classes, the more predicted masks for each RoI. This allows Mask R-CNN to decouple classification and mask prediction. Only the mask associated with ground-truth class is used for calculating the loss.

### Mask Representation

A $m \times m$ mask is predicted for every single RoI using an FCN. This fully convolutional representation requiers fewer parameters and is more accurate than using just *fc* layers.

### RoIAlign

The RoIPool of Fast R-CNN and Faster R-CNN first quantizes a floating-number RoI to discrete granularity. This qunatization produces small misalignments which have a large negative effect on pixel-accurate masks.

To fix this, the authors created RoIAlign. This layer removes harsh quantization of RoIPool. To do this, they simply remove any rounding used. Instead, they use bilinear interpolation tocompute the exact values of the input features at four regularly sampled locations in each RoI bin, and aggregate the result.

### Network Architecture

For the *backbone* architecture, ResNet 50 and 101, ResNeXt 50 and 101, and Feature Pyramid Networks (FPN) were tested.

For the *heads*, they extend two existing Faster R-CNN heads. One using the fourth stage of ResNet, the other using a FPN.

## 3.1 Implementation Details
