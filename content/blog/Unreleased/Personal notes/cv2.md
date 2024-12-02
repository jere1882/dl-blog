---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: OpenCV - cv2 library
title: cv2
description: NOtes on cv2
date: 2024-11-26
image: /thumbnails/backbones.png
---
The `cv2` library is part of OpenCV (Open Source Computer Vision Library), a popular open-source computer vision and image processing library. It provides tools for **analyzing, processing, and manipulating images** and videos. 

### **Key Families of Functionality** 
These functions represent specific areas of functionality in OpenCV: 
#### 1. **Thresholding and Binarization** 
`cv2.threshold`, `cv2.adaptiveThreshold` - Converts grayscale images to binary based on a threshold value. The former uses the same global threshold and the later calculates thresholds dynamically for smaller regions of the image.
#### 2. **Morphological Transformations**
Helps in cleaning up binary images, filling gaps, or separating objects. Operates on binary images to remove noise or to emphasize specific shapes.

 `cv2.erode`, `cv2.dilate`
 `cv2.morphologyEx` : Applies advanced morphological transformations like opening, closing, gradient, etc., to process and clean up binary images.
 * Opening: Erosion followed by dilation (removes small noise)
 * Closing: Dilation followed by erosion (fills small gaps in objects)
 * Gradient: Highlights the edges of objects
#### 3. **Contour Detection and Analysis** 
Used for extracting and analyzing object boundaries.
`cv2.findContours`: Contours are curves joining continuous points with the same intensity (e.g., boundaries of objects).
 `cv2.drawContours`: Draws contours on images.
`cv2.contourArea`: Computes the area of a contour
`cv2.approxPolyDP`: Approximates a contour with a simpler polygon, reducing the number of points. Uses the Ramer-Douglas-Peucker algorithm to reduce the number of points in a curve while preserving its overall shape. The `epsilon` parameter controls the degree of approximation (lower values = closer approximation). Used for shape simplification (e.g., converting curves into polygons), object detection, and boundary simplification.
`cv2.arcLength`: Computes the area of a contour
`cv2.isContourConvex`: Check if a contour is convex. Convex contours are simpler and are often used to filter shapes.

#### 4. **Geometric Transformations and ROI (Region of Interest) Extraction**

Object localization, ROI extraction, and preparing image data for machine learning.

`cv2.boundingRect`: Computes the smallest upright rectangle that bounds a contour. Assumes an axis-aligned bounding box
`cv2.convexHull`: Generates the convex boundary of a contour.
`cv2.minAreaRect`: Returns the smallest rotated rectangle that bounds a contour. - 
`cv2.minEnclosingCircle`: Finds the smallest circle enclosing a contour
 
#### 5. **Pixel-Level Statistics**
Enables statistical analysis of images or specific regions

 `cv2.mean`: Computes the mean and standard deviation.
`cv2.meanStdDev`: Counts non-zero pixels in a binary image.