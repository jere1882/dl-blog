---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: a-review-of-object-detection-architectures
title: A Review of Object Detection Architectures
description: In this post I explore the most popular model architectures used to tackle object detection
date: 2024-08-30
image: /thumbnails/pick_architecure.jpeg
---
## Introduction

After spending several days understanding the details of the popular YOLO family of object detection architectures, I decided to write my own summary and notes, mostly with references to other handy articles about object detection architectures.

Let's start by defining **object detection** as a computer vision task where you try to predict bounding boxes containing objects within an input image, from a predefined set of interesting objects.

![[Pasted image 20240830215401.png]]

This is one of the most fundamental tasks in computer vision, one step up in complexity to image classification and one step down to image segmentation. Recent advances in deep learning have yielded extraordinarily accurate object detection models with countless practical applications.

![[Pasted image 20240830220153.png]]

In this post I will summarize and discuss the most important architectures for tackling object detection with deep learning.

## Overview of Architectures

Object detection architectures can be broadly grouped into two main categories: **two-stage detectors** and **one-stage detectors**.

### Two-stage detectors

**Two-stage detectors** first generate region proposals where objects might be located, then refine these proposals and classify the objects within them. These methods **prioritize accuracy over speed**.

| **Architecture** | **Strengths**                               | **Weaknesses**                         | **Use Cases**                       |
|------------------|---------------------------------------------|----------------------------------------|-------------------------------------|
| **R-CNN**        | High accuracy, foundational approach        | Very slow, computationally expensive   | Research, detailed image analysis   |
| **Fast R-CNN**   | Faster than R-CNN, shared computation       | Still slower than modern methods       | Applications where accuracy is key  |
| **Faster R-CNN** | High accuracy, end-to-end training          | Slower than one-stage detectors        | Tasks requiring precise localization|
| **Mask R-CNN**   | Adds instance segmentation, very accurate   | Computationally intensive              | Object detection + segmentation     |
### One-stage detectors
One-stage detectors perform object detection in a single pass, directly predicting class probabilities and bounding boxes from the input image. These methods are faster but may sacrifice some accuracy compared to two-stage detectors.

| **Architecture** | **Strengths**                               | **Weaknesses**                         | **Use Cases**                       |
|------------------|---------------------------------------------|----------------------------------------|-------------------------------------|
| **YOLO**         | Real-time detection, fast and efficient     | Less accurate for small objects        | Real-time applications, robotics    |
| **SSD**          | Good speed-accuracy trade-off, multi-scale  | Less precise than two-stage detectors  | Mobile applications, embedded systems |
| **RetinaNet**    | Focal Loss improves detection of hard cases | Slower than YOLO, complex training     | Detection in challenging scenarios  |
I will pick one architecture for each category to delve deeper into the details: YOLO and Faster R-CNN.
## YOLO

YOLO, which stands for "You Only Look Once," is a widely recognized real-time object detection algorithm that has become a cornerstone in the field of computer vision. First introduced in 2016, YOLO was revolutionary for its speed, outperforming all other object detectors available at the time.

Since the original release, YOLO has undergone several iterations, each bringing significant advancements in performance and efficiency. Let's see an overview of the different versions:

| **YOLO Version** | **Main Innovations**                                                                 | **Authors / Research Group**               | **Release Date** |
| ---------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ | ---------------- |
| **YOLO**         | First real-time object detection algorithm; single-stage detection.                  | Joseph Redmon et al.                       | 2016             |
| **YOLOv2**       | Improved accuracy, multi-scale detection, anchor boxes.                              | Joseph Redmon et al.                       | 2016             |
| **YOLOv3**       | Multi-scale predictions, better performance on small objects, Darknet-53 backbone.   | Joseph Redmon and Ali Farhadi              | 2018             |
| **YOLOv4**       | CSPDarknet backbone, mosaic data augmentation, self-adversarial training.            | Alexey Bochkovskiy, Chien-Yao Wang, et al. | April 2020       |
| **YOLOv5**       | Improved usability, PyTorch implementation, auto-learning bounding boxes.            | Ultralytics                                | June 2020        |
| **YOLOv6**       | Faster inference, more compact model architecture, optimized for edge devices.       | Meituan                                    | 2022             |
| **YOLOv7**       | Efficient Layer Aggregation Networks, advanced model scaling, cross mini-batch norm. | Chien-Yao Wang, Alexey Bochkovskiy, et al. | July 2022        |
| **YOLOv8**       | Enhanced usability, ease of training and deployment, more accurate and faster.       | Ultralytics                                | January 2023     |
| **YOLOv9**       | Further speed and accuracy improvements, enhanced feature extraction techniques.     | Ultralytics (anticipated future version)   | February 2024    |
### YOLO Step by Step

## Misc

### Benchmarks and metrics
The most popular benchmark is the [Microsoft COCO dataset](https://cocodataset.org/). 

![[Pasted image 20240830223641.png]]

Different models are typically evaluated according to a Mean Average Precision (MAP) metric. It combines the concepts of precision and recall across different object classes and detection thresholds.
* **Precision**: The proportion of true positive detections (correctly detected objects) among all positive detections (both true and false positives).
* **Recall**: The proportion of true positive detections among all actual objects (true positives plus false negatives).
* **Average Precision** is the area under the Precision-Recall curve for a specific class.
* **mAP** is the mean of the Average Precision (AP) values computed for each class in the dataset. It provides a single metric that summarizes the model's performance across all classes.



Great references:

https://viso.ai/deep-learning/object-detection/
