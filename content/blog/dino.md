---
has_been_reviewed: false
tag: Computer Vision, Foundation Models
tags:
  - Foundational-models
aliases: 
publish: true
slug: dino
title: DINO - self-distillation with no labels
description: Let's go over the paper "Emerging Properties in Self-Supervised Vision Transformers"
date: 2024-10-27
image: /thumbnails/DINO.jpeg
---
# Introduction

Let's go over the 2021 paper that introduced the popular approach DINO:
![[Pasted image 20241027200558.png]]
From Meta (Facebook), this paper presents a technique to pretrain Vision Transformers on large volumes of unlabelled image data called. Its successor, DINOv2, is widely used in many cutting-edge foundation multimodal models and it's considered a SOTA method for pretraining models with image inputs.
# Recap: The ViT architecture

Let's quickly revisit the ViT architecture, which I have described in depth in other posts. Most specifically, let's focus on the usage of the `[CLS]` token:
![[Pasted image 20241027200910.png]]

The `[CLS]` token is a single embedding added to the embeddings of all the patches obtained from the image. Within the encoder of the transformer, it is treated as any other image patch, going over successive transformer blocks that perform self attention across all patch embeddings.

At the end of the final transformer layer, the embedding corresponding to the `[CLS]` token has interacted with all the patches of the image via self attention, and can be considered to contain a summary information of the entire input image.

Many downstream tasks, such as image classification, would discard all the output of the transformer encoder except for the attention-augmented vector corresponding to the `[CLS]` patch. Thus, we can append a MLP to it and make image classification.

# Recap: Self Supervised Learning (SSL)

- A major trend in DL is to train models in huge datasets of unlabelled data using pretext tasks: this is called SSL.
- These tasks enable the model to discover and learn rich, meaningful representations of the underlying data structure on its own.
- Once these powerful representations are learned, they can be applied to a wide range of downstream tasks , with minimal or no need of annotated data.
- These representations form the backbone of what we now refer to as foundation models—large, pre-trained models that can be adapted to a wide range of downstream tasks.
- In the image domain, pretext task may involve masking areas of the image and tasking a network to fill them in.

# Recap: Knowledge Distillation

Let’s formally define knowledge distillation, a technique traditionally used for model compression:
![[Pasted image 20241027223623.png]]
Knowledge distillation is elegantly defined in section 3.1 of the paper:

Knowledge distillation is a learning paradigm where we train a student network `g_{θ_s}` to match the output of a given teacher network `g_{θ_t}`. parameterized by `θ_s` and `θ_t` respectively.

Given an input image x, both networks output probability distributions over K dimensions denoted by `P_s` and `P_t`. The probability `P` is obtained by normalizing the output of the network `g` with a softmax function. 

Given a fixed teacher network `g_{θ_t}`, we learn to match these distribution by minimizing cross-entropy loss w.r.t. the parameters of the student network `θ_s`:

`min_{θ_s}   -P_t(x) * log(P_s(x))`

# Emerging Properties in Self-Supervised Vision Transformers

## Attention maps

This paper (and its successor, DINOv2) provide a state-of-the-art method to train ViT on unlabelled images.

The ViTs trained with this method are able to learn what parts of the image are really worth attending to, as shown by the many attention maps of the CLS token shown in the paper: 

![[Pasted image 20241027201717.png]]
*The paper shows a lot of attention maps. These maps come from the very last transformer block in the encoder. Given the CLS token embedding, self-attention will calculate `k_{CLS} * q_{p}` for all embedding patches `p`. This product will produce  a single scalar value for each patch `p` . We can reassemble the patches back into a 2D image, and color each patch with this attention coefficient. That 's  what this images show, basically how much the CLS token output attended to each of the different patches. Notice that these guys use tiny patch sizes.*

## Why are these attention maps remarkable?

Without any supervision, the authors are able to attend to the important parts of an image. In fact, a sort of semantic segmentation emerges from the attention maps.

The model learns about what areas are more informative to featurize or characterise an image. This rich representation has not been achieved by previous SSL setups.

![[Pasted image 20241027223459.png]]

## Combining self supervised learning and knowledge distillation

DINO combines the idea of knowledge distillation with self supervised learning with images. The learning process is going to involve a student and teacher network.

### Multi-crop strategy

![[Pasted image 20241027224359.png]]
- Let's assume there is a dataset of unlabelled images 
- Given each image, we generate local crops and pass them through the **student** network.
- Global crops of the same image are passed through the **teacher** network.
- We calculate cross entropy loss between the student and teacher output, and update the student weights via regular gradient descent. In other words, the student tries to match the teacher output.
### The student and the teacher

In traditional knowledge distillation, the teacher is a given complex network that was trained on labelled data. Here however, the teacher is not given. We fabricate it! 

The student and the teacher are neural networks with identical architecture (e.g. a ViT or a ResNet + a projection head), parameterized by weights `theta_s` and `theta_t`.

The student weights are optimized via gradient descent using cross entropy loss against the teacher outputs, which encourages it to learn from the teacher.

The teacher weights are an exponential moving average of the student weights. In other words, the teacher is built out of past iterations of the student

![[Pasted image 20241027225205.png]]
Lambda has a schedule from 0.99 to ~1. This is called a “momentum encoder” teacher.

Why is this good at all? Why would this “slow” teacher be any good to train the student learning?

Turns out that it has a bunch of desirable properties:
- The EMA approach can be seen as a way of “model ensembling”, which is known to boost performance
- The fact that we feed “better crops” makes the teacher make better predictions, just becase it has more information. This induces a "local to global" learning.
- The teacher consistently has better validation performance than the student, therefore it guides the training by providing higher quality features
- Centering and scaling of the teacher output prevent collapse
![[Pasted image 20241027225344.png]]
If you have studied other SSL techniques for images, notice that only positive pairs are used, unlike other methods that require negative pairs.

### Architectures

As mentioned before, the architectures of the teacher and student are identical.  Also note that the student and teacher ultimately produce a vector of k features each, describing an image.

The architectures explored were:

- ResNet50 and ViT (ViT/S, ViT/B). For ViT, they explore also different patch sizes (usually set to 16, but they insist on the advantages of using 8 and 5)
- They plug a MLP with a single hidden layer at the end of either the ResNet or the embedded `[CLS]` token for CNN and ViT respectively  which produces the final K features. 

## DINO: Summary 

The main features of DINO are:
* Multi Crop Strategy →  Encourages local to global correspondences, gives the teacher more context that the student needs to infer
* Momentum Encoder teacher → Guides the training of the student providing high quality reference features
* Centering and sharpening → Prevent collapse
* Non discriminative approach, cross entropy loss
* Works great with ViT

# Experiments and Results

## How SSL methods are evaluated

How do you use DINO (or any SSL trained model) for image classification? There are different options:
1- Fully unsupervised: k-NN. Store the features of each training sample, and given a new sample, find the closest matches in your feature space and predict their average label.
2- Linear: Freeze the network, add linear layers at the end of the features and train them using annotated data. 
3- Fine Tuning: Similar but unfreeze the network.  

## Evaluation of DINO

The paper makes a very thorough evaluation of this method. I will highlight a few results.

### Image Classification - ImageNet

DINO+ViT beats all other SSL image approches in ImageNet using both 20-kNN and Linear approaches. Moreover, it almost reaches the performance of a dedicated, supervised learning trained method.

Furthermore, if we unleash the ViT complexity, DINO reaches #1 spot in ImageNet beating all supervised and self-supervised learning methods (at a great computational expense though)

Figure 11 shows how meaningful are the embeddings by plotting a 2D-projection. Notice how images of similar concepts cluster together:

![[Pasted image 20241027225937.png]]

### Image Retrieval
Image retrieval task: Given an image and a database of images, find the closest match.
- Oxford and Paris image ; and also Google Landmark v2 retrieval datasets
- query/database pairs
- DINO features outperform all other architectures trained on ImageNet labels.

![[Pasted image 20241027230028.png]]
 
### Experiments and results: Copy Detection
 
- INRIA Copydays Dataset
- The task is to recognize images that have been distorted by blur, insertions, print&scan, etc
- DINO+ViT outperforms other reported methods

![[Pasted image 20241027230100.png]]

### Experiments and results: Instance Segmentation

They don’t train any layers on top of the originally trained feature extractor. Apparently they just threshold the self-attention maps to keep 60% of the (attention) mass. Their goal is not to do actual segmentation but just to show how their representation encodes meaningful semseg data.

![[Pasted image 20241027230121.png]]

# Beyond 2021's DINO

https://dinov2.metademolab.com/

DINOv2 (2023) enhances DINO by:

- Using flash attention, a novel attention mechanism that is “faster and better”
- Nested tensors in self attention, bringing substantial improvements in computational efficiency
- Fully-Sharded Data Parallel (FSDP) : Facilitate training of huge models like ViT-g in multiple GPUs
- Allows use a ViT-g teacher and then distill knowledge for smaller student models instead of training from scratch

And they demoed it in a bunch of other tasks without any supervision, and you can try your own images in their website!

# Research use case: AstroCLIP (2024)

AstroCLIP Is a multimodal foundation model for galaxies. It embeds galaxy images in different bands as well as spectrum into the same latent space, then use the latent space representation for downstream tasks.

DINOv2 is used to pretrain the galaxy image embedding in a HUGE unlabelled dataset of 76M galaxy images. These are not traditional “rgb” images, they are instead images where each channel represents a different band.

![[Pasted image 20241027230254.png]]