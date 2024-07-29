# ViT

The whitepaper that put forth the Vision Transformer (ViT), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, was published in June 2021. This paper is the origin of using Transformers for Computer Vision (CV).

## Abstract

The Transformer had been the "standard for natural langauge processing tasks" for a few years at this point, but it had yet to be applied to computer vision tasks. In this paper, the Google Brain Team showed that not only could Transformers be used for Computer Vision tasks, but that they could beat CNNs in both performance and computational resoures required for training.

## Introduction

Pre-training a self-attention/Transformer based model on large amounts of text and then fine-tuning for task-specific problems has become standard in NLP. Inspired by recent successses, the authors experminted with using Transformers on CV tasks. To do this, they split images into patches and used these patches as the sequences for the Transformer. So, instead of word-piece embeddings, they used image patch embeddings. In this paper, the authors trained on an image classification problem in a supervised fashion

On "mid-sized datasets" like ImageNet, these models achieved nearly the same accuracy as ResNets of similar size. While this might at first seem discouraging, it is not suprising given Transformer's lack of inductive biases inherent to CNNs (translation equivariance and locality). On larger datasets, however, this pattern does not stand. Essentially, large scale training trumps inductive bias.

## Related Work

Naive application of self-attention to images would have each pixel being used as its own token. This would mean each pixel attends to every other pixel. Given that this is quadraticly expensive with respect to the nubmer of pixels, this doesn't scale well for even small images. Using small patches of pixels is much more efficient. Some work has attempted this as well, including one where the patches are 2x2.

## Method

Model design as similar to original Transformer as possible. The input to the model is a 1D sequence of token embeddings. 
- To handle 2D images, the images are reshaeped into a sequence of flattened 2D patches known as the patch embeddings. 
- Similar to BERT's `[class]` token, a learnable embedding is prepended to the sequence of patch embeddings, whose state at the output of the Transformer encoder serves as the image representation $y$. 
- A one hidden layer MLP is attatched to the final `[class]` output during pre-training, and by a single linear layer during fine-tuning.
- Position embeddings are added to the patch embeddings to retain positional information. These are standard learnable 1D positional embeddings (since the 2D aware positional embeddings don't seem to improve performance)
- The Transformer encoder consists of alternating layers of multiheaded self-attention (MSA) and MLP blocks. 
- Layernorm (LN) is applied before every block, and residual connecitions after every block. 
- The MLP contains two layers with a GELU non-linearity.
