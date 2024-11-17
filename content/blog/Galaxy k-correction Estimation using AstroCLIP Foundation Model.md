---
tags:
  - Computer
  - Vision
aliases: 
publish: true
slug: k-correction-via-foundation-model
title: Galaxy k-correction Estimation using AstroCLIP Foundation Model
description: Fine tune an astronomical foundation model to tackle a novel task
date: 2024-11-12
image: /thumbnails/astronomy.jpeg
---
# Introduction

**Foundation models** are transforming machine learning, setting new benchmarks across industries by leveraging their ability to learn patterns from huge unlabelled datasets. Earlier this year (2024), [**AstroCLIP**](https://arxiv.org/abs/2310.03024) was introduced—a multimodal foundation model trained on millions of galaxies. AstroCLIP maps both galaxy images and spectra into a shared, richly expressive k-dimensional embedding space, enabling a wide range of applications from redshift estimation to morphology classification.

Building on AstroCLIP’s capabilities, I set out to tackle a novel challenge: predicting **galaxy k-corrections** directly from galaxy images. K-corrections are critical for deriving intrinsic galaxy properties, but traditional methods are only able to provide gross estimates if spectroscopic data is not available. 

This project bridges the gap between astrophysics and modern machine learning by developing a method that extends AstroCLIP to predict k-corrections. To achieve this, I experimented with multiple approaches. For **few-shot learning**, I explored fine-tuning AstroCLIP's embeddings by attaching models like multi-layer perceptrons (MLPs) and more sophisticated networks. For a simpler **one-shot approach**, I used k-nearest neighbor (kNN) regression directly within AstroCLIP’s embedding space. 

These methods allowed me to harness the rich latent representations learned by the foundation model, avoiding the need for costly end-to-end training on the full dataset.

To my knowledge, this is the first approach to accurately predict k-corrections from galaxy **images** alone. My model achieved a $R^2$  score of 0.8 on a 200,000-galaxy dataset from DESI-EDR, surpassing the SOTA estimates calculated by Blanton's method. This project highlights the power of combining foundation models with tailored machine learning techniques and underscores the role of innovative ML solutions in advancing astrophysical research.

![[Pasted image 20241115155237.png]]
# Data and task definition

Astronomy is a field rich with data, and modern surveys like [DESI](https://data.desi.lbl.gov/) generate vast datasets from countless celestial objects. In this project, we'll focus on galaxies—one of the most fascinating and complex objects in the universe—and explore how machine learning can help address a long-standing challenge in astronomy: calculating K-corrections.

In this section, I will to present at a very high level the astronomical problem I'm trying to solve in this project. You may skip this section if you don't care about the astronomy involved.
## Understanding Galaxy Measurements

Galaxies can be studied through three primary types of measurements:

- **Spectra:** These are detailed measurements of a galaxy's light broken down by wavelength. They are highly informative, although expensive and sometimes impossible to obtain.
- **Images in different bands:** These capture the structure of a galaxy across specific ranges of wavelengths (e.g., optical or infrared bands), providing valuable spatial information.
- **Photometry:** This refers to the total light collected from a galaxy in each band, offering a simpler but less detailed view than spectra.

![[Pasted image 20241116192853.png]]
***Images in different bands** of Galaxy NGC 5055* 

![[Pasted image 20241116193115.png]]*A galaxy **spectrum** (black line) with three overlaid band response functions (colored curves). Each band has an associated **photometric measurement**. *
## What is the K correction?

A K-correction is a crucial value in astronomy that adjusts for the effect of the universe's expansion on the light we observe from galaxies. For a given band (e.g., `R`), the K-correction `K_{R,R}` accounts for how the redshift alters the observed brightness in that band. If multiple bands are of interest (e.g. g, r, z), we calculate a corresponding K-correction for each.

When spectra are available, K-corrections can be calculated analytically. This is neatly derived in [Hogg's 2002 paper](https://arxiv.org/abs/astro-ph/0210394):

![[Pasted image 20241116193727.png]]

However, spectra are often unavailable. In such cases, astronomers rely on photometry-based methods like [Blanton's K-correction](https://arxiv.org/abs/astro-ph/0606170), which, while standard, can be inaccurate—particularly for distant galaxies.

## Leveraging ML for K-corrections

This is where machine learning can make a significant impact. By leveraging a foundation model pretrained on 76 million images and spectra, we can develop a novel method to estimate accurate K-corrections directly from galaxy images. Since images are far more accessible than spectra, this approach has the potential to become an invaluable tool for astronomers working with large datasets.

For this project, I used a subset of data from the [DESI Early Data Release](http://data.desi.lbl.gov/doc/releases/edr/), which includes the following quantities for about 200,000 galaxies:

* Astonishingly high-resolution spectra (wavelengths from 3600 A to 9824 A sampled at 0.8 intervals). The spectra is encoded as a `1x7780` array.
* Images in DESI bands g, r and z. For each galaxy, the images are cropped and represented as a `144x144x3` array.
* Redshift `z`, a single Real number for each galaxy.

While this dataset has been instrumental in proving the concept, I am already working on extending the project to another survey that offers even higher-resolution data.
# Data visualization and preparation

## Inspecting the original dataset

The original dataset, downloaded from DESI-edr release and exported by AstroCLIP authors in hugging face format, has exactly 197976 galaxy samples, split into train (0.8) and test (0.2) datasets.

In this notebook you can find the code I used to visualize the data.

This is what the spectrum and images of an individual sample in our dataset look like:

![[Pasted image 20241116201730.png]]
It is informative to inspect the distribution of the column redshift (`z`) across our galaxies:

![[Pasted image 20241116235811.png]]
This gives us an idea of the distance of the galaxies we are working with. The bulk of the galaxies are at `z <= 0.6`, which can be considered "nearby" at cosmological distances. Only a handful (~800 samples) are at redshift `>1`.

As a final sanity check, let's validate that the redshifts are equally balanced across the training and test split:

![[Pasted image 20241117015818.png]]

## Calculating the Target Labels

The original DESI data release does not provide K-corrections. However, since our the release includes galaxy spectra, we can calculate K-corrections analytically using Hogg's equation.

In [this notebook](http://localhost:9999/notebooks/data/AstroCLIP/calculate_k_corrections.ipynb), I implemented Hogg's method, which integrates the spectra to calculate K-corrections in any desired band. 

The ultimate goal of this project is to predict K-corrections **directly from galaxy images**. The values calculated using Hogg's equation will only serve as the target labels to fine-tune a machine learning predictor.

For the purposes of this blogpost, I will focus on analysing the K correction at band `sdss_r` with band shift 0.1. That said, the distribution of our target variable looks like this:

![[Pasted image 20241117004541.png]]

It is also informative to visualize the distribution of the target variable discriminating by galaxy redshift.

![[Pasted image 20241117021053.png]]
Looking at this plot reveals a clear discontinuity at z ~ 0.38. Interestingly, this is caused by the [Balmer jump](https://en.wikipedia.org/wiki/Balmer_jump), and shows us that for galaxies beyond z ~ 0.38 the K correction at band `sdss_r` is very hard of impossible to predict.

Therefore, for the purposes of this blogpost, let's restrict our dataset to galaxies whose z lies to the left of the 0.38 discontinuity.

# Extending AstroCLIP foundation model

Having properly calculated and cleaned our labels, we are finally able to gauge the power of foundation models for this task.

The AstroCLIP foundation model is huge, involving an image encoder with 307M parameters and a spectrum encoder with 43M trainable parameters. The results of either encoder gets processed by further cross-attention heads and MLP, outputting a final embedding of size 512 for any galaxy (regardless of whether the input was an image or a spectrum).

For the purposes of this project, we can treat AstroCLIP as a black box that simply converts galaxy images into a very rich 512-dimensional vector. Indeed, by loading the AstroCLIP model checkpoint, I leveraged the trained model and mapped every galaxy image from my dataset into a 512-dimensional vector.

The task now becomes to predict the target k-correction from the 512-D featurised representation of each image. There are several approaches we can take here, let's explore them from the simplest to more complex.
## One-shot learning

A straightforward approach is to apply K-nearest neighbors in the latent space. This does not require training any additional layers on top of AstroCLIP and works surprisingly well: 0.025 mean absolute error and 0.87 `R^2` score for using number of neighbors k=64.

![[Pasted image 20241117025006.png]]
![[Pasted image 20241117045212.png]]


*This section will be extended soon showing the effect of the number of neighbors in the R^2 score, and how the value 64 was picked.*

However, this approach has a few key disadvantages, inherited from KNN approach:
1- Poor generalization beyond training data.
2- Inference scales poorly with the size of the training dataset, and this can be particularly harmful in large astronomical datasets.

## Few-show learning

A more interesting approach involves adding extra trainable layers at the end of AstroCLIP.
### Simple MLP layers

Adding a few dense layers with ReLU activation and Dropout is a popular and simple way to fine tune foundation models for specific tasks. A single hidden layer with 64 units, plus a single output neuron for regression achieves 0.10 MAE and 0.86 `R^2` score. This is slightly worse than the KNN approach.

In my experiments so far, I found that increasing the number of hidden nodes or hidden layers doesn't have a significant impact.

### Ongoing work
Expect this section to be updated soon! I'm exploring:
* Outputing a not just a single value, but a mean and variance ; in order to be able to tell the model confidence on its prediction.
* Appending attention layers to the end of AstroCLIP, not just a MLP.
# Benchmarking

For benchmarking pusposes, I used [k-correct 5.1.2](https://kcorrect.readthedocs.io/en/5.1.2/), a tool that estimates K-corrections based on galaxy photometry rather than spectra. While this method is less accurate, it is the standard approach when spectra are unavailable, making it an ideal baseline for comparison with the ML model's predictions I am presenting.

I will plot a comparison between k-correct estimations and my own ML model estimation soon.
