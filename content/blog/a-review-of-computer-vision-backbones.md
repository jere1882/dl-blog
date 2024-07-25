---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: a-review-of-computer-vision-backbones
title: A review of backbones used in computer vision
description: In this article I survey different backbone alternatives used in modern computer vision architectures
date: 2024-07-20
image: /thumbnails/backbones.png
---
## Introduction

In my previous [post](link), I introduced the task of semantic segmentation along with introductory experiments on the benchmark dataset Cityscapes.

This post will dig deeper into the different architectural choices for semantic segmentation, describing their building blocks, advantages and shortcomings.

To make this post more interesting, I decided to train and validate all the architectures here described on a relatively new dataset, so that we can see actual performance stats for a particular instance: XXXXXX-Medical-Dataset.

The following table provides an overview of the architectures that will be developed in this post:

| **Architecture**                           | **Year** | **Main Innovation**                                                                                                    | **Reference Paper**                         |
| ------------------------------------------ | -------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **FCN (Fully Convolutional Networks)**     | 2015     | First architecture to use fully convolutional layers for segmentation.                                                 | Long, Shelhamer, and Darrell, CVPR 2015     |
| **U-Net**                                  | 2015     | Symmetric encoder-decoder structure with skip connections.                                                             | Ronneberger, Fischer, and Brox, MICCAI 2015 |
| **SegNet**                                 | 2015     | Efficient skip connections that only share pooling indexes.                                                            | Badrinarayanan et al., 2015                 |
| **DeepLab v1**                             | 2015     | Introduced atrous convolutions to control resolution.                                                                  | Chen et al., ICLR 2015                      |
| **PSPNet (Pyramid Scene Parsing Network)** | 2017     | Pyramid pooling module to capture global context.                                                                      | Zhao et al., CVPR 2017                      |
| **DeepLab v2**                             | 2017     | Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.                                                         | Chen et al., PAMI 2018                      |
| **DeepLab v3**                             | 2017     | Enhanced ASPP with global pooling and wider range of dilations.                                                        | Chen et al., arXiv 2017                     |
| **DeepLab v3+**                            | 2018     | Added decoder module to DeepLab v3 for better boundary refinement.                                                     | Chen et al., ECCV 2018                      |
| **ViT**                                    | 2020     | Applied transformers to computer vision tasks. This is not actually used for semantic segmentation but paved the road. | Dosovitskiy et al., 2020                    |
| **SegFormer**                              | 2021     | Transformer-based architecture for segmentation.                                                                       | Xie et al., NeurIPS 2021                    |
| **Swin Transformer**                       | 2021     | Hierarchical Transformer with shifted windows for efficient modeling.                                                  | Liu et al., ICCV 2021                       |

## The Encoder-Decoder 

With the exception of the transformer based methods, the overwhelming majority of semantic segmentation architectures follow an **encoder-decoder** structure:

![[Pasted image 20240605063627.png]]

In this setup, the network is split in two main components:

 The **encoder** (aka **backbone**) extracts features from the input image through convolutional layers. 
 * Standard encoders used for other tasks, such as image classification, can be used here. VGG and ResNet are popular choices, and are often pre-trained on generic datasets such as ImageNet.
 * The output of the encoder is a feature representation of the input image in the form of an input map. A typical encoder could, for example, take as input an image of shape (256,256,3) and output a feature map of shape (16,16,512).
 * A typical encoder, as shown in the example above, chains convolutions and pooling layers, reducing the spatial dimension of the image significantly while extracting high level features.
 
The **decoder** takes the encoded feature representation, and produces a segmentation mask.
* Usually upsampling layers and convolutions are used to restore the original resolution of the image
* Many architectures employ skip connections, which greatly aids to recover the fine details lost during the contracting encoder path.

At first, the encoder-decoder design seemed counter-intuitive to me. Why would we want to "compress" the data only to later on "expand" it? Why would we want to introduce this bottleneck in the middle? It turns out there are many good reasons to use this pattern:

1. **Hierarchical feature extraction**: The encoding process allows the network to capture and condense essential features and context from the input data at multiple scales. 
2. **Global context**: By progressively downsampling, the network can aggregate information over larger regions of the input image, capturing global context which is often critical for making accurate predictions. For example, recognizing that a specific region is part of an organ might require understanding the broader context of surrounding regions.
3. **Spatial Understanding:** When the data is upsampled in the decoder, this global context is combined with high-resolution features from the encoder via skip connections, facilitating precise localization and segmentation.
4. **Efficient Computation:** Compressing the data reduces the dimensionality, making the computations more efficient. This can lead to faster training and inference times, and also reduces the memory requirements.
5. **Noise Reduction:** The bottleneck can act as a form of implicit regularization by filtering out noise and less important details, thus helping the network to generalize better from limited data.

## Fully Convolutional Network (FCN)

Original publication: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) - Long, Selhamer, Darrell.
_MMsegmentation model zoo_: [link to repo](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/fcn/README.md)

FCNs were a breakthrough architecture for semantic segmentation tasks. While not the absolute first, they were one of the earliest and most influential approaches to leverage deep learning for this purpose.

FCNs replaced fully-connected layers with convolutional layers. This allowed them to process images of varying sizes and output a pixel-wise classification, matching the input image dimensions.


![[Pasted image 20240606055901.png]]
*Illustration from the original paper*

The encoder path progressively downsamples the input image via convolutions and pooling. Any backbone could be used here (VGG, ResNet, etc);  the choice also depending on the desired size of the network.

In the picture, we can see an encoder that progressively reduces the spatial dimension, while increasing the number of channels. The final output of the encoder is a feature map with 4096 channels.

The decoder then applies 1x1 convolution layers to project the high-level features from the encoder to a number of channels equal to the number of classes you want to segment (in this example, 21). 

The original FCN relied on simply upsampling the feature maps in the decoder path to recover spatial information. This approach, however, could lead to a loss of fine-grained details. Later architectures like U-Net and DeepLab incorporated skip connections to address this issue.

## UNet

_Original paper:_ [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597) -  Olaf Ronneberger, Philipp Fischer, and Thomas Br 
_Original use case:_ Biomedical image segmentation

![[Pasted image 20240607055016.png]]

Quoting the very clear explanation of the paper:

*The network architecture (...) consists of a contracting path (left side) and an expansive path (right side). The **contracting path** follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions, each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels.* 

*Every step in the **expansive path** consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. (...) At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.*

The crucial innovation here is the introduction of **skip connections** from the contracting path to the expansive path. They connect the encoder and decoder layers with the same resolution, allowing the network to use both the high-resolution features from the downsampling path and the upsampled features in the upsampling path. This helps to recover spatial information that might be lost during downsampling.

✅ High performance, able to produce fine grained predictions.
❌ Sharing entire feature maps as skip connections involve high use of memory and slower training.
## SegNet

 Original Paper: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation - Vijay Badrinarayanan, Alexander Kendall, Animabadi Ramesh

![[Pasted image 20240607060140.png]]

In the paper, the authors describe how the encoder part is quite standard:

*The encoder network in SegNet is topologically identical to the convolutional layers in VGG16. We remove the fully connected layers of VGG16 which makes the SegNet encoder network significantly smaller and easier to train than many other recent architectures*

The key difference with UNet is that skip connections do not share entire feature maps. Instead, each time that max pooling takes place on the encoder side, the indexes where the max feature was located are stored and conveyed to the appropriate decoder layer. This is much more memory efficient.

 *(...) storing only the max-pooling indices, i.e, the locations of the maximum feature value in each pooling window is memorized for each encoder feature map. In principle, this can be done using 2 bits for each 2 2 pooling window and is thus much more efficient to store as compared to memorizing feature map(s) in float precision.*

The decoder progressively upsamples the image in a peculiar way. It makes use of the memorized max-pooling indexes to produce sparse maps, which are in then convolved to produce dense (non sparte) feature maps:

![[Pasted image 20240607060812.png]]
*This image compares the upsampling procedure of SegNet (left), in contrast to the standard deconvolution used by other models like FCN and UNet*


✅ SegNet method of sending pooling indexes to the decoder is more **memory efficient** and involves less data movement during training, resulting in **faster training**.
❌ Pooling indexes inherently discard some spatial information that may not be recoverable, leading to less precise localization in segmentation tasks.

## PSPNet

Original paper: "Pyramid Scene Parsing Network" by Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. - 2017

The core innovation of PSPNet is the Pyramid Pooling Module. This module is designed to capture contextual information at multiple scales. It achieves this by applying pooling operations at several grid scales, such as 1x1, 2x2, 3x3, and 6x6. These pooled features are then upsampled to the original size and concatenated together to form a robust global feature representation.

![[Pasted image 20240720043714.png]]

The first step in a PSPNet is the CNN block, quoting the paper:

*Given an input image, we use a pretrained ResNet model with the dilated network strategy to extract the feature map. The final feature map size is 1/8 of the input image*

Then, the novel Pyramid Pooling Module comes in:

*The pyramid pooling module fuses features under four different pyramid scales. The coarsest level highlighted in red is global pooling to generate a single bin output. The following pyramid level separates the feature map into different sub-regions and forms pooled representation for different locations. The output of different levels in the pyramid pooling module contains the feature map with varied sizes. To maintain the weight of global feature, we use 1×1 convolution layer after each pyramid level to reduce the dimension of context representation to 1/N of the original one if the level size of pyramid is N. Then we directly upsample the low-dimension feature maps to get the same size feature as the original feature map via bilinear interpolation. Finally, different levels of features are concatenated as the final pyramid pooling global feature. 

Finally, a convolution layer is used to generate the final prediction map in (d).

The key innovation is pooling at different scales,which is essential for capturing features that vary in size and context within an image. E.g. 1x1 pooling captures global context by summarizing the entire feature map into a single representation ; whereas larger-scale pooling (e.g.2x2, 3x3, 6x6) captures local context at different granularities.

✅ Pyramid Pooling Module is great at understanding global context of the scene, which is crucial for accurate segmentation. Achieves SOTA results on several benchmark datasets.
❌ Can be slower and more resource-intensive compared to architectures like SegNet, U-Net, or FCN. The implementation is also more complex.

## DeepLab

This is actually a family of architectures designed and open-sourced by Google. Each version enhances the previous one, here's an overview:

- DeepLab v1: Introduced **Atrous (aka Dilated) convolution** as well as **CRF Post-Processing.** CRF was like a beautification step applied at the end, not affecting the loss and not trained – dropped in v3.
- Deeplab v2: Introduced enhanced use of atrous convolutions by the **Atrous Spatial Pyramid Pooling (ASPP)** + stronger backbone like ResNet.
- DeepLab v3: Furter improved the ASPP + batch normalization + deeper backbones
- DeepLab v3+

Perhaps the central innovation of v1 was the introduction of **atrous (aka dilated) convolutions** instead of pooling layers.
* Atrous convolution inserts spaces (or zeros) between the filter weights, effectively "dilating" the filter. As a consequence, the convolution has a larger receptive field, skipping values in between.
* Unlike pooling or strided convolutions, atrous convolution does not inherently change the spatial dimensions
* They are very useful for semantic segmentation, where maintaining spatial dimensions and fine details is crucial. Thus, they offer a simple and powerful alternative to deconvolutions. You don't reduce the spatial resolution in the first place, so you don't need to upsample it.


![[Pasted image 20240720045947.png]]
*Atrous convolutions*

![[Pasted image 20240720050524.png]]
*How atrous convolutions are comparable to pooling+regular convolution*

The **Atrous Spatial Pyramid Pooling (ASPP)** introduced in v2 has a resemblance to PSPNet's Pyramid Pooling Module. It applies multiple dilated convolutions with different dilation rates, capturing multi-scale context by probing the incoming feature map with filters at multiple sampling rates. This enables the network to capture both local and global context effectively.

![[Pasted image 20240720050124.png]]

The next version, v3, further improves the architecture by:
- Enhancing the ASPP with image-level global features and more dilation rates
- Removing the CRF post-processing
- Using a deeper backbone and batch normalization
- Introducing the **atrous depth-wise separable convolution**, which basically increases computational efficiency. It's basically a depth-wise convolution followed by a point-wise (1x1) convolution.

Finally, v3+ incorporates a decoder module to refine the segmentation results, particularly object boundaries:
* The encoder remains similar to v3: A backbone (e.g. ResNet) followed by the ASPP module.
* The newly added decoder refines the segmentation output by upsampling and concatenation with corresponding low-level features from earlier layers of the encoder (like the skip connections of the U-Net)

![[Pasted image 20240720051926.png]]