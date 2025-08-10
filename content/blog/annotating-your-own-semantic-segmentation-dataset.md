---
tag: Computer Vision
aliases: 
publish: true
slug: annotating-your-own-semantic-segmentation-dataset
title: Annotating your own Semantic Segmentation dataset
description: In this blog post, I explore the challenges and tools involved in creating a semantic segmentation dataset for underwater scenery.
date: 2024-06-18
image: /thumbnails/annotators.png
---
## Introduction
In my recent project "semantic segmentation of underwater scenery", I decided to create my own semantic dataset of underwater scenes [1]. I used pictures I took with my GoPro during a diving vacation in French Polynesia and the Red Sea. My initial goal was to assemble a dataset of a few hundred images with reasonably accurate masks.

It took me 2 hours to annotate my first 4 samples, painstakingly drawing polygons and outlines of small fish and corals. Over time, I became more accustomed to the annotation interface and discovered a few tricks. However, this process has been a humbling experience.

Before this project, I always took dataset annotations for granted, rarely considering the immense complexity associated with data annotation. In this post, I will outline the tools that helped me ease the process, and showcase the process step by step.

## Different tasks and their annotations

Image classification, object detection, and semantic segmentation are increasingly challenging computer vision tasks, and their corresponding annotations also grow in complexity:

![compvision tasks](/assets/compvision_tasks.png)
* Annotating samples for image classification is as simple as looking at an image and typing a label.
* Object detection usually involves drawing bounding boxes and associating a class label with each box. This is more time-consuming but arguably still manageable if the number of objects per image is relatively small.
* Semantic segmentation annotations, on the other hand, can be incredibly complex. It involves assigning a class label to each pixel on the image. This can be done by drawing polygons, shapes, using a brush to paint areas, etc..

Take the following image from my SUIM dataset as an example:

![Pasted image 20240618221315](/assets/Pasted%20image%2020240618221315.png)
*For the curious reader, this is Rangiroa Island in French Polynesia*

Annotating the boundary of the coral reef and the ocean, as well as the outlines of the many small fish visible, is a daunting task.

> 💡 **Remark:** Never underestimate the importance of high quality annotations. Your models are going to be only as good as your data is.
## Annotating data with CVAT

[CVAT](https://www.cvat.ai/) (short for Computer Vision Annotation Tool) is a great tool that I discovered after struggling to use several alternatives. It offers a free version for individuals and small teams, and the graphical user interface (GUI) is intuitive and comprehensive.

I'll walk you through the process of annotating an image in CVAT.

1. **Register in CVAT:** You need to sign up and login to CVAT At [https://app.cvat.ai/](https://app.cvat.ai/). We'll use the tool's web interface.
2. **Create a new project**: CVAT uses a hierarchical structure: projects contain tasks, and tasks contain jobs.

The second step is to create a new project. Go to the Projects tab, click the "+" symbol, choose a project name, and list all the labels your segmentation model will recognize.

This is what my underwater segmentation project looks like after adding all the labels:

![Pasted image 20240618222027](/assets/Pasted%20image%2020240618222027.png)

3. **Create a new task** - within the newly created project. Here, you'll be asked to provide a name for the task and, most importantly, upload the images you want to annotate.
4. **Create a new Job** - under the recently created task. Here, you can use the default settings, which primarily control the order in which the images are presented for annotation.
5. **Open the new Job, and start annotating.** You will be presented with a first image to annotate, such as this one:
![Pasted image 20240618222150](/assets/Pasted%20image%2020240618222150.png)*A school of fish, 30 meters deep in the Tuamotu Islands*

The annotation tools are available on the left panel:

* **Pentagon icon**: This lets you draw polygons by clicking on key points that define the outline of an area of a certain class (e.g., a fish). The polygon will be filled with the chosen class label. Click "done" after each individual annotation.
- **Brush:** The brush can be set to different stroke sizes and is useful for drawing detailed boundaries.
- **Circles and squares:** These shapes are occasionally useful, although complex boundaries are more common in semantic segmentation.
- **Magic wand:** We'll discuss this tool in the next section, as it's quite helpful!

![Pasted image 20240618222645](/assets/Pasted%20image%2020240618222645.png)

Notice that all your annotations for a given image are listed on the right hand side. Annotations have a "z" integer value that determines how overlapping structures are displayed. For instance, if two polygons overlap, the one with the higher "z" value is supposed to occlude (cover) the one with the lower value.

![Pasted image 20240618223215](/assets/Pasted%20image%2020240618223215.png)
*Objects on an annotation, sorted by Z value. You can modify the z value by using "send to background" and "send to foreground" actions.*


> 💡 **Tip:**  It's often helpful to draw a square covering your entire image and assign it the class that represents the background (e.g., "Sea Floor/Rocks" or "Background"). Set this background object to have a lower z-value. This way, all areas not covered by other objects will default to the appropriate background class.

I use a combination of these tools depending on what works best for each individual annotation. Once you're finished, click the save button on the top panel. Then, press the forward icon to move on to the next image.
## AI tools

CVAT offers AI tools to help you speed up the annotation process significantly. Currently, there are two options: "Segment Anything" and "EfficientViT Annotator." These tools suggest annotation boundaries, which you can refine by indicating points inside and outside the objects.

![ia tools annotation](/assets/ia_tools_annotation.gif)
*From CVAT blogpost*

While not perfect, these AI tools can be a huge time saver. When trying to use them, I often encountered imprecise boundaries and needed to use the brush and polygon tools for refinement. However, they are generally accurate, so I always recommend trying them before resorting to manual annotation.

This section covers everything you need to know to get started creating your first annotated image set for semantic segmentation! The following sections provide some background details for the curious reader.
## The cost of annotations

Several companies specialize in providing professional data labeling services, including Appen, HIVE, and CloudFactory. Additionally, tech giants like Google and Amazon offer their own annotation services. The cost of these services can vary depending on the complexity of the task and the turnaround time required.

For instance, [Google Data Labeling Service]([https://cloud.google.com/ai-platform/data-labeling/pricing](https://cloud.google.com/ai-platform/data-labeling/pricing)) offers tiered pricing based on the volume of data annotated. The following table shows the price per unit per human labeler, with "Tier 2" pricing applied after exceeding 50,000 annotations.\

![Pasted image 20240621192013](/assets/Pasted%20image%2020240621192013.png)
A semantic segmentation mask has many “units” (for simplicity, let's think of units as polygons). So, my understanding is that an it costs e.g. $0.87 per polygon. Or if you need this task performed by multiple annotators (presumably for robustness), the cost would be multiplied by the number of annotators per image.

Say we want to annotate an underwater image with 10 fish, plus a big coral reef. That would be 11 polygons. 11 units = 9.35 USD. That value, times the number of annotators per image (which may be 2-3) ; would be the final value of annotating a single image. (around $28).

As you can see, semantic segmentation data is crazily expensive to annotate, and the cost scales up with the number of individual structures within the image.

> 💡 **Fact:**  According to Indeed.com, the average salary of a professional annotator in the US is approximately $42 per hour (for June 2024).
## Segment Anything. How does it work?

[Segment Anything](https://segment-anything.com/) (SAM) is an AI model developed by Meta that can automatically segment objects in images, even if those objects are unfamiliar. This is particularly impressive because traditional semantic segmentation models often struggle with unseen categories.

![[section-1.1b.mp4]]

The 2023 paper (“Segment Anything”)[[https://arxiv.org/pdf/2304.02643](https://arxiv.org/pdf/2304.02643)] unveils the details of this great tool. The paper highlights the creation of a new dataset specifically designed for SAM:

_We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images._

That’s massive. 11M images, imagine the cost and effort of getting those annotations! Notice that they refer to a “a new task”. More specifically, it introduces the concept of a "promptable segmentation task," allowing users to interact with the model by providing points within an image. The model then leverages this information to segment the corresponding object, even if it's a novel category.

Let's see what the authors said about the model itself:

_A powerful image encoder computes an image embedding, a prompt encoder embeds prompts, and then the two information sources are combined in a lightweight mask decoder that predicts segmentation masks. We refer to this model as the Segment Anything Model, or SAM_
![Pasted image 20240623082253](/assets/Pasted%20image%2020240623082253.png)
Let's dig a little more into the key components of this architecture. About the **image encoder**, the paper says:

_we use an MAE pre-trained Vision Transformer (ViT) minimally adapted to process high resolution inputs. The image encoder runs once per image and can be applied prior to prompting the model._

Let's break this down:
* The ViT part is a ViT-H/16, which means a vision transformer with standard 16x16 patches, 14x15 windowed attention and four attention blocks.
* Masked autoencoder (MAE) refers to a crucial pre-training step. A portion of the image patches are randomly hidden, and the ViT only sees the unmasked patches. It gets tasked with reconstructing the original complete image. This process forces the ViT to learn robust representation by relying on context and relationships between visible patches.

The second key component is the **prompt enoder**, of which the authors said:

_We consider two sets of prompts: sparse (points, boxes, text) and dense (masks). We represent points and boxes by positional encodings summed with learned embeddings for each prompt type and free-form text with an off-the-shelf text encoder from CLIP. Dense prompts (i.e., masks) are embedded using convolutions and summed element-wise with the image embedding_

Finally, the **Mask Decoder** combines the encoded image and encoded promt, to generate segmentation masks. According to the authors:

_employs a modification of a Transformer decoder block followed by a dynamic mask prediction head. Our modified decoder block uses prompt self-attention and cross-attention in two directions (prompt-to-image embedding and vice-versa) to update all embeddings. After running two blocks, we upsample the image embedding and an MLP maps the output token to a dynamic linear classifier, which then computes the mask foreground probability at each image location_

I encourage you to read [this great post]([https://medium.com/@utkarsh135/segment-anything-model-sam-explained-2900743cb61e](https://medium.com/@utkarsh135/segment-anything-model-sam-explained-2900743cb61e)) to understand the fine details. 
## Conclusion

In this blog post, we explored the world of semantic segmentation annotations. We saw the challenges of manual annotation, the tools that can ease the process (like CVAT), and the high costs associated with professional services. We then peeked into the exciting world of AI-powered annotation tools, specifically Meta's "Segment Anything" model.