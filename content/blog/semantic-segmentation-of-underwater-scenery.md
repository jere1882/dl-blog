---
tags:
  - Computer
  - Vision
aliases: 
publish: true
slug: semantic-segmentation-of-underwater-scenery
title: Semantic Segmentation of Underwater Scenery
description: Application of ViT and other architectures to a novel semantic segmentation task.
date: 2024-07-24
image: /thumbnails/underwater_segmentation.png
---
## Introduction

Embarking on the journey to become a proficient machine learning engineer requires more than just theoretical knowledge; it demands hands-on experience with real-world challenges. Over the past four months, I have undertaken a comprehensive project entirely on my own, which I am thrilled to present in this blog post.

The focus of this project is **semantic segmentation of underwater scenery**, which involves the delineation of various semantic elements within diving photographs, such as fish, corals, vegetation, rocks, divers, and wrecks. 

Inspired by the 2020 paper "[Semantic Segmentation of Underwater Imagery](https://arxiv.org/pdf/2004.01241)," I began by meticulously cleaning and correcting the publicly available dataset from this research. Recognising the need for a more robust and diverse dataset, I extended it with new images from my own diving trips, investing significant effort to manually annotate them in detail. I published this [enhanced dataset](https://www.kaggle.com/datasets/jeremiasrodriguez/segmentation-of-underwater-scenery-extended) at Kaggle.

To tackle the task of underwater semantic segmentation, I employed both classical architectures used by the original paper and cutting-edge visual transformer architectures. Over the course of this project, I trained six different large models, and I am pleased to have produced models that surpass the benchmark results established by the original paper. Don't miss the demo videos at the end of the post!

Join me as I delve into the details of this project.
## Dataset and Task

In essence, the task involves assigning a class to each pixel in an input image. 

![[Pasted image 20240716181748.png]]

The eight classes defined in the original paper, which I have adhered to, are:

![[Pasted image 20240715154933.png]]

The underwater nature of these images introduces a host of unique challenges:

1. **Variable Visibility**: Water composition and weather conditions significantly impact visibility. Depth also plays a critical role, as colors, especially reds, fade the deeper you go.

![[Pasted image 20240715171157.png]]
*Demonstration of color degradation at different depths - [Source]([https://youtu.be/AAJjdA6b4Ts](https://youtu.be/AAJjdA6b4Ts))*

2. **Diverse Natural Landscapes**: Aquatic environments vary dramatically by geographic location. A good model must learn high-level concepts, such as the general appearance of fish, as it is impractical to provide training samples for all existing species.

![[Pasted image 20240715171445.png]]
*This chart showcases common species in French Polynesia, from [Franko Maps](https://frankosmaps.com/products/tahiti-fish-id).* Notice the wide variety of shapes, colors and patterns.

3. **Complex Scenery**: Underwater landscapes can be difficult to interpret even for humans. Many animal species have evolved to blend seamlessly with their surroundings, making it challenging to accurately identify and delineate its shape. Classes such as `reefs and invertebrates` can't often be distinguished from `sea-floor and rocks`, as many corals often look just like rocks and the sea-floor is often covered in dead coral. 

> 💡 **Why is this task relevant?** Semantic understanding of underwater scenery is crucial for autonomous underwater robots. By comprehending their surroundings, these robots can navigate more effectively, identify and monitor marine life, map underwater environments, and carry out complex tasks such as inspecting underwater structures or assisting in search and rescue operations. This advanced understanding enhances their ability to perform autonomously with higher precision and safety in diverse and challenging underwater conditions.

In order to solve this task, we cast this problem as a supervised learning semantic segmentation task. The authors of the original paper published a dataset of approximately 1600 labelled images, which I have corrected and extended using photographs of my own diving trips.

Let's look at a few of the new images that I introduced to the dataset, and discuss the challenges they pose in the semantic segmentation task:

![[clownfish2.png]]
*A cute anemonefish hiding in an anemone. A good model should be able to understand that the fish belongs to the `fish` class whereas the anemone is an `invertebrate`.*

![[Diodon hystrix(porcupinefish)2 1.png]]
*A porcupinefish hiding in the shadows, easily overlooked.*

![[razor_coral2.png]]
*Certain species of corals can be easily confused with vegetation, such as the famous "lettuce coral"*

![[Sixbar wrasse2.png]]
*A complex underwater scene involving `aquatic plants`, `fish`, `sea floor`, `reefs` and `bodywater` classes. A good model has the challenge to understand the many actors involved in this scene.*

![[Pasted image 20240715172954.png]]
*A clever octopus, blending in with the coral in the background.*

![[sea_turtle3.png]]
*A sea turtle at 30 meters depth. The loss of color at great depths is very evident in this picture. A good model should be able to handle color degradation.*

In order to enhance the pre-existing dataset, I did the following:

1. Identified 86 masks in the original training set that needed corrections, and re-annotated them from scratch.
2. Selected 166 images from my diving trips to French Polynesia and the Red Sea and manually annotated their segmentation masks.

The resulting dataset has exactly 1801 annotated images, which I published at [Kaggle](https://www.kaggle.com/datasets/jeremiasrodriguez/segmentation-of-underwater-scenery-extended). Annotating data was a very humbling experience for me, since I used to underestimate how time consuming and complex it can be. I wrote a separate [blogpost tutorial](annotating-your-own-semantic-segmentation-dataset) explaining how you can annotate your own dataset, as well as a few insights and tricks.
## Experimental setup

I had two goals in mind when I undertook this project. Firstly, reproducing the results from the original paper. Secondly, trying to improve their results by using my new extended dataset and leveraging cutting-edge architectures like visual transformers.

I therefore selected three standard architectures:
* [DeepLabv3+](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus)
* [PSPNet](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/pspnet)
* [UperNet](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/upernet)
As well as two vision transformer architectures:
* [(Regular) ViT](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/vit/README.md)
* [Swin ViT](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/swin/README.md)

Which I trained on both the original dataset, as well as in my enhanced version. Training took a few days for each model using 4 GPUs and 160k iterations, and was therefore quite time consuming. 

I leveraged the toolbox [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/vit/README.md) (checkout my tutorial on [this blogpost](mmsegmentation-tutorial)!), which already provides high quality implementations of the architectures I mentioned. Check the implementation details on my GitHub repo, since I had to make a few adjustments to the code to support the new task. Pre-trained weights (from ImageNet) were used for all models.

| Model       | Backbone       | Crop size  | Number of trainable parameters |
| ----------- | -------------- | ---------- | ------------------------------ |
| DeepLab v3+ | ResNet50       | 512 x 1024 |                                |
| PSPNet      | ResNet50       | 512 x 1024 |                                |
| UperNet     | ResNet101      | 512 x 512  |                                |
| Swin ViT    | not applicable | 512x512    |                                |
| ViT         | not applicable | 512x512    |                                |

The original paper reports results on two similar architectures (DeepLabv3 and PSPNet), which I will use as a baseline to compare the performance of my model.

| Model     | Backbone    | Resolution | Number of parameters |
| --------- | ----------- | ---------- | -------------------- |
| DeepLabv3 | unspecified | 320x320    | 41.254 M             |
| PSPNet    | MobileNet   | 384x384    | 63.967 M             |
| SUIM-Net  | VGG         | 320x256    | 12.219 M             |
*From table III of the SUIM paper. SUIM-Net is an original architecture proposed by the authors, which achieved the best results on their paper.*
## Experiment results

### Results of the original paper (Baseline)

Recall the list of classes defined by the original authors:
![[Pasted image 20240715154933.png]]
The authors reported results **only** for classes HD, WR, FV, RI and RO. I imagine they did this because (1) PF is quite under-represented in the dataset and performance is terrible on it and (2) class SR is visually almost indistinguishable from class RI.

The following table summarises the results reported the original paper:

| Model      | HD    | WR    | FV    | RI    | RO    |
| ---------- | ----- | ----- | ----- | ----- | ----- |
| PSPNet     | 75.76 | 86.82 | 74.67 | 85.16 | 72.66 |
| DeepLab v3 | 80.78 | 85.17 | 79.62 | 83.96 | 66.03 |
| SUIMNet    | 85.09 | 89.90 | 83.78 | 89.51 | 72.49 |
*IoU of five classes, as reported in table II of the original paper.*

### Replication of the results of the original paper

I trained two of these models using the exact same dataset that they used, in order to replicate their results. I decided to report stats on all eight classes, although only the first five columns allow for numerical comparison with the paper results:

| Model      | HD    | WR    | FV    | RI    | RO    | SR    | VG    | BG    | mIoU  |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| PSPNet     | 85.39 | 73.13 | 75.08 | 73.11 | 73.32 | 67.62 | 12.11 | 88.87 | 68.59 |
| DeepLabv3+ | 86.33 | 69.48 | 76.79 | 72.46 | 81.91 | 65.68 | 12.06 | 88.68 | 69.17 |
| UperNet    | 86.18 | 72.2  | 77.53 | 70.35 | 81.88 | 66.39 | 6.73  | 88.61 | 68.73 |

 💡 **Interpretation of results:** If we make a pairwise comparison between the original paper results and my own replication, looks like my models are significantly better at segmenting HD, RO and FV ; whereas their models do better in the under-represented class WR, and in the umbrella class RI. I believe the divergence is due to the fact that they seem to have merged classes RI and SR for training as well as different model parameter choices. 
### New architectures and extra data

In my last set of experiments, I attempted to surpass the results presented so far by:

1. Adding 181 extra images to the training dataset, collected and annotated by myself.
2. Trying vision-transformer based architectures, which have proven very successful in many computer vision tasks in recent years.

The results are summarised in the following table:

| Model       | HD    | WR    | FV    | RI    | RO    | SR    | VG    | BG    | mIoU  |
| ----------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| PSPNet      | 85.28 | 74.8  | 76.34 | 74.49 | 81.35 | 68.24 | 8.77  | 88.96 | 69.78 |
| DeepLab v3+ | 86.06 | 72.55 | 75.18 | 71.28 | 77.93 | 65.23 | 13.35 | 88.2  | 69.72 |
| UperNet     | 85.85 | 70.39 | 78.9  | 73.36 | 84.57 | 65.41 | 10.5  | 88.53 | 69.69 |
| Swin ViT    | 84.47 | 71.2  | 85.46 | 72.35 | 86.27 | 67.2  | 9.63  | 88.56 | 71.14 |

**TL;DR:** The extra data gives all models a boost of ~1 point of mIoU. Moreover, the SWIN ViT is able to surpass all CNN-based architectures by an extra ~1.5 mIoU.


![[Pasted image 20240719061154.png]]
*Training loss of Swin ViT. The loss function decreases quickly in the first 20k iterations, and gradually plateaus.*

## Demo of the best model: Inference over diving videos

To showcase the performance of the best model and also their limitationes, I hand picked a couple of clips from my last diving vacation and used the best trained model to perform inferences over each frame.

Let's have a look at the results: 

<iframe width="916" height="389" src="https://www.youtube.com/embed/EOsqx5n4w60" title="Semantic segmentation of Underwater Scenery - Playful dolphin demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
*A playful dolphin during one of my dives in Rangiroa - The segmentation is quite decent taking into account that there is just a single dolphin in the entire training dataset. The model manages to identify the dolphin as "fish and vertebrate" when the camera captures its entire body, but it struggles to differentiate it from the diver when the dolphin is partially occluded.*

<iframe width="916" height="389" src="https://www.youtube.com/embed/2MQDcxpr5B8" title="Sharks and Rays in Bora Bora - Semantic Segmentation demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
*A snorkelling session in Bora Bora, showcasing a huge school of black-tip sharks, stingrays, as well as humans snorkelling (me!) and smaller fish. This hectic scene poses a challenge for the model, which is trained in less populated scenarios and hasn't seen any humans snorkelling, only divers. Yet, the segmentations are quite resonable, showing great generalization power to unseen scenarios. Notice the sea floor segmentation often flickering between `reef` and `sea-floor and rocks` classes.*

## Conclusion and Future Work

In this project, I tackled the task of underwater semantic segmentation. I began by replicating the results of the 2020 paper "[Semantic Segmentation of Underwater Imagery](https://arxiv.org/pdf/2004.01241)," successfully training three CNN-based models that achieved IoU values comparable to those reported by the paper.

I then extended the public dataset by adding 181 original images and annotations. Training the same models on the extended dataset resulted in an increase of about 1 point in mIoU. Additionally, I trained a modern SWIN ViT architecture, which was developed after the original paper was written. Using ViT led to an increase of 1.5 points in mIoU.

Overall, the results are encouraging and the segmentations are reasonable, as demonstrated in the demo videos. It is important to note that the dataset used is quite small by modern deep learning standards, containing only 2000 samples. Despite this, the trained models are able to produce sensible segmentations, which would be bound to get better if more annotated data was made available.

One challenge encountered was distinguishing between `Sea floor and Rocks` and `Reefs and Invertebrates`, with the models often oscillating between these classes. The authors of the original paper seemed to have merged these classes under the hood, and my experiments could benefit from a similar approach, as they are often indistinguishable and mislead the training process.

The models also struggled to produce precise boundaries for divers and fish that blend with their surroundings. This may be due to imprecise annotations in the dataset and the small size of the training dataset in general.

To sum up, I am proud of the models' ability to identify varying species of fish and human divers, which are two of the most interesting and interactive classes for potential applications. I enjoyed refining my computer vision skills, and I hope the public dataset continues to grow.