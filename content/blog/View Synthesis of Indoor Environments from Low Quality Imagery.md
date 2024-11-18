---
tags:
  - Computer
  - Vision
aliases: 
publish: true
slug: indoor-nerf-reconstruction
title: View Synthesis of Indoor Environments from Low Quality Imagery
description: Personal project where I train and compare NERF models on indoor images
date: 2024-09-17
image: /thumbnails/NERF.png
---
# Introduction

In my [previous post](https://www.jeremias-rodriguez.com/blog/introduction-to-nerfs), I explored Neural Radiance Fields (NeRFs)—a remarkable technique for synthesizing images from unseen viewpoints, given a training set of images with annotated camera poses.

This time, I’m taking the theory into practice by training several NeRF variants on images of my own **60 m² apartment**. You’ll see detailed reconstructions of rooms and objects, demonstrating the impressive capabilities of these methods.

To push the limits of NeRFs, I imposed significant constraints on the training data. I restricted the camera poses to lie in a plane parallel to the floor (e.g., **z = 5 cm** for one dataset and **z = 1.60 meters** for another) and reduced the resolution of the original iPhone images to just **720x480**.

These challenges make the task much harder. NeRFs are typically tested on small objects and often struggle to scale to larger, more complex scenes. Additionally, training on images from a single plane makes it especially difficult to reconstruct views at different camera heights, and lower image resolution exacerbates the problem. Yet, the strong results I obtained despite these restrictions highlight NeRFs’ potential for applications in **robotics industries**, where constrained data collection is a common scenario.

Finally, I’ll showcase advanced NeRF variants that incorporate NLP prompts to perform tasks requiring semantic understanding of the environment. As a bonus, I’ll compare NeRFs with an emerging alternative: **Gaussian Splatting**.

# Data Collection

Let's quickly go over the two datasets that I collected for these experiments. Below you can see a plane of my apartments, where I collected video using an iPhone 14.

* For dataset 1, I walked around holding the phone in my forehead, capturing video for about 3 minutes from different viewpoints. I was always facing forward, meaning that the camera poses should roughly lie at `z=1.75` meters and a fixed angle around the `x` and `y` axis.
* For dataset 2, I attached the phone to a wheeled toy at a height of 5 cm from the floor. I moved the toy around the home, capturing footage from a very low viewpoint. Again, there was no rotation along the `x` and `y` axis. I did zoom out to get a fisheye effect and better expose higher areas of the room.

After extracting the individual frames from each video, the resulting datasets are

| Dataset   | Height  | Number of Images | Camera settings     | Original Resolution |
| --------- | ------- | ---------------- | ------------------- | ------------------- |
| Dataset 1 | ~175 cm | 4980             | Standard iphone 14. | 1080x1920           |
| Dataset 2 | ~5cm    | 8823             | 0.5x zoom.          | 1080x1920           |

I downgraded the quality of the images to 540x960 in order to test the ability of the models to reconstruct scenes from low quality imagery, and also to lower memory footprint. Moreover, due to GPU constraints, for many methods I had to downsample the frame-rate of dataset itself.

# Camera Pose Estimation

This is the trickiest step in the process, and perhaps the most critical one. For most NERF methods to work, we need a training dataset of images, each one annotated with their 3D camera position. Intrinsic camera parameters are also required.

For the purposes of this post, let's assume that no prior knowledge of camera poses is available, as is the case of my iphone datasets. In an upcoming post, I will be discussing how to integrate prior pose knowledge, e.g. from a SLAM system.

[**Colmap**](https://colmap.github.io/) is the standard tool to be used to estimate the poses of a set of overlapping images, and I found that most NERF implementations assume people have run Colmap on their data before training.

The pose estimation step is in my opinion quite undocumented, and colmap can be **very finicky**. Often fails without any hint of what is wrong, so I will make an effort to explain how it works step by step, and share debugging techniques I learned along the way. You can skip the next section if you are not interested in the pose construction steps.

A good, clean dataset is the single most important factor to a good NERF result:

* If possible, remove featureless images such as closeups of walls.
* Try to expose the scene from as many viewpoints as possible.
* If possible, remove blurry images.
* Try to prevent too many dynamic objects, if possible have none.
* For the current SOTA methods, keep your dataset size bounded, don't go over 10K, images, ideally much less.
* For the current SOTA methods, don't try to reconstruct environments larger than `~100 m^2`
* Consistent good lighting
* If you have too many duplicates or very high frame-rate, consider downsampling
* Reconstruction quality degrades significantly with low image resolution
* Make sure all images were obtained with the same camera, using the same settings ; and if possible research the camera model used.
## Step 1: Feature Extraction

Ensure you first install the last version of colmap:

`sudo apt install colmap`

The first step is **feature extraction**, and involves detecting and describing keypoints (distinctive features) in each input image. These keypoints serve as the foundation for matching and estimating the relative positions of cameras and 3D points in the scene

Assuming that input images are stored in `imgs` folder, feature extraction can be run as follows:

```
colmap feature_extractor --database_path database.db --image_path imgs/ --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1 --ImageReader.camera_model 
```

The parameters provided are:
* **--database_path**:  A location where a new database will be created, and the results of feature extraction (and subsequent steps) will be stored. **Ensure that you remove an older database that may live in this same path.**
* **--image_path**: A folder storing your training images
* **--ImageReader.single_camera**: We are forcing the model to assume all the images come from the same camera. This is not suitable if you change camera settings on the fly or if you are using images from different cameras.
* **--SiftExtraction.use_gpu 1**: Speed up using GPU. Later steps are actually going to take forever unless you use a GPU.

If you happen to know that your camera uses a specific [camera model](https://colmap.github.io/cameras.html), and if you even happen to know the intrinsic parameters of your camera, then you should definitely input them to the feature extractor, e.g by adding these parameters to the command:

```
--ImageReader.camera_model OPENCV_FISHEYE --ImageReader.camera_params "281.08,282.98,319.51,236.08,0.01,-0.23,-0.27,0.19"
```

For the iPhone images, I used the default camera model. 

The execution time of this step is very short, less than 2 minutes. After it finishes, colmap should have populated several tables on database.db. I strongly encourage you to sanity-check the database by inspecting it:

```bash
$ sqlite3 database.db 
SQLite version 3.31.1 2020-01-27 19:55:54
Enter ".help" for usage hints.
sqlite> select count(*) from images;
8823
sqlite> select * from images limit 5;
1|1000.jpg|1
2|10002.jpg|1
3|10000.jpg|1
4|10001.jpg|1
5|10003.jpg|1
sqlite> select * from descriptors limit 5;
1|576|128|
2|462|128|
3|504|128|
4|449|128|
5|470|128|
)&
```

There should be as many images in the `images` tables as there were in your folder, and both the `.descriptors` and `.keyframes` tables should have been populated. There should also be one entry in `.cameras` table.

## Step 2: Feature Matching

In this computationally expensive step, colmap is going to try to match the features extracted in the previous step across multiple images. There are several strategies that can be used to reduce computation time, but I have opted for **exhaustive matching** which consistently ensures best results. Exhaustive matching scales very poorly to large image datasets, it can take hours for datasets with up to 10000 images, and I strongly encourage you to seek alternatives (such as **sequential matching**) if your dataset is larger.

Exhaustive matching can be run as follows:

```
colmap exhaustive_matcher --database_path database.db --SiftMatching.use_gpu 1
```

After it finishes, the table `.matches` should have been populated in your `database.db`

## Step 3: Sparse Model Generation

The third and last step is to geometrically verify the matches and iteratively estimate the camera poses:

```
colmap mapper --database_path database.db --image_path imgs/ --output_path sparse --Mapper.ba_global_function_tolerance=1e-5
```

The parameter `--Mapper.ba_global_function_tolerance=1e-5` governs one of the termination conditions, and I suggest trying `1e-6` to `1e-4` values depending on the quality of your data and the execution time you are willing to endure. This step can take several hours if you impose a very low tolerance and/or if your dataset is too large.

The results of this step will be, finally, estimations of your camera poses, stored in the subfolders of the `sparse` folder. Often, colmap will produce several alternative estimations, and store them in subfolders named `sparse/0`, `sparse/1`, `sparse/2`, etc.

The best one is supposed to be `/0`, however I have often found that `/0` has been able to reconstruct the pose for very few points, and I end up using an alternative such as `/1`. 

Let's discuss how to inspect and debug one of these reconstructions, e.g. `/0`. Inside one such folder, you should find three files: 

* **cameras.bin**: Here are listed all the cameras (we actually constrained the problem to just one camera, so just one entry). For each camera, you'll find the estimated intrinsic paramaters.
* **images.bin**: Here are the estimated poses of each image that colmap managed to map. Notice that a few of the original images, sometimes the vast majority of them, may not have entries here meaning that colmap failed to estimate their pose.
* **points3D.bin**: Points in the scene that colmap has seen across several images, and their triangulated pose.

Actually, in order to read the contents of these files, I encourage you to convert the bin files into txt files as follows:

```
colmap model_converter --input_path /sparse/0 --output_path /sparse/0 --output_type TXT
```

If your ultimate goal is to train NERFs, the most important part is to check the fourth line of images.txt, which may say something like this:

`# Number of images: 8725, mean observations per image: 497.11243553008597`

This indicates the number of images whose pose has been estimated. Aim at this number being a significant proportion of your original dataset, ideally `>90%`. If your dataset has too many featureless frames, I've seen this number degrade quickly, even to values as low as `3%` and the output model only matching featureless frames amongst themselves.

If `sparse/0` has too low a proportion of poses estimated, checkout `sparse/1`, since previous iterations of the mapper may have arrived at decent reconstructions.

Colmap has a GUI that you can use to load and visualize a sparse model. I will be showcasing NeRFStudio visualizer later on though, which is much better to visualize your estimated camera poses.

## Step 4: Refine Poses

Optionally, you may try to refine your camera pose and intrinsics estimates even further. In my experience this can run for too long and produces unnoticeable results, but I suppose it must be helpful in some situations because systems like NeRFStudio do this by default:

```
colmap bundle_adjuster --input_path sparse/0 --output_path sparse/0 --BundleAdjustment.refine_principal_point 1
```


# Training NeRFs

After exploring a handful of implementations, I came across [NeRFStudio](https://docs.nerf.studio/) , which conveniently implements a wide variety of cutting-edge NeRF methods:

* It works perfectly with Viser in order to visualize training.
* Intuitive API for training, rendering videos.
* Supports tracking with W&B.
* Well documented and maintained.

The star method implemented by NeRFStudio is **nerfacto**, which basically takes the best ideas of recent innovations in the field and fuses them into a single implementation. 

Here is a short demo of one of the NeRF's I trained at home. Stay tuned while I finish writing this blogpost!


<iframe width="916" height="389" src="https://www.youtube.com/watch?v=DoU7-JTeMzw" title="NeRF of my apartment" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>




