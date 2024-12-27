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
The `cv2` library is part of OpenCV (Open Source Computer Vision Library), a popular open-source computer vision and image processing library. `cv2` are actually python bindings to C++. It provides tools for **analyzing, processing, and manipulating images** and videos. 

### **Key Families of Functionality** 

These functions represen`t specific areas of functionality in OpenCV: 
#### Read and Write

- `cv2.imread(path, flag)`: Reads an image. Flags: cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED.
- `cv2.imwrite(path, img)`: Saves an image to a file.
- `cv2.imshow(winname, img)`: Displays an image in a window.

note: imshow is tricky, it blocks your screen. Make sure you call it like this:

```python
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Basic manipulation
- `resized_img = cv2.resize(img, (width, height), interpolation)`
- crop an image via slicing: `cropped_img = img[y1:y2, x1:x2]`
- `new_img = cv2.cvtColor(img,code)`: color conversion cv2.`COLOR_BGR2GRAY`, `cv2.COLOR_BGR2RGB` 
- `rotated_img = cv2.rotate(img, rotateCode)` with `code cv2.ROTATE_90_CLOCKWISE`
- `mod_img = cv2.warpAffine(img, M, dzise)` where M is a `2x3` transformation matrix and dsize is the output image. It preserves lines and parallelism, but not necessarily distances and angles.

Move the image 50 pixels to the right and 30 pixels down:

```python
import cv2
import numpy as np

# Load the image
img = cv2.imread("image.jpg")

# Define the translation matrix
M = np.float32([[1, 0, 50], [0, 1, 30]])

# Apply warpAffine
translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

cv2.imshow("Translated", translated)
cv2.waitKey(0)
```
Rotate the image 45degrees around its center

```python
rows, cols = img.shape[:2]

# Compute the rotation matrix
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Center, angle, scale

# Apply warpAffine
rotated = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
#### Drawing (inplace)
- `cv2.line(img, pt1, pt2, color, thickness)`
- `cv2.rectangle(img, pt1, pt2, color, thickness)`

## Filtering
1. `cv2.GaussianBlur(img, (ksize, ksize), sigmaX)`: Applies Gaussian blur to an image. It reduces noise and smoothens the image by applying a Gaussian kernel.
- img: The input image. It can be a grayscale (2D) or color (3D) image. Data type: numpy.ndarray, typically uint8 or float32.
- (ksize, ksize): Kernel size (width, height). Must be odd (e.g., (3, 3), (5, 5)).
- sigmaX: Standard deviation in the X direction. Determines the spread of the Gaussian kernel. If 0, it is calculated based on ksize.
Returns a blurred version of the input image. Data type and dimensions are the same as the input image.

2. `cv2.Canny(img, threshold1, threshold2)`: Detects edges using the Canny edge detector. It basically calculates gradients and based on that determines the edge locations.

Inputs:
- img: Input image. Must be a single-channel grayscale image. Data type: numpy.ndarray, typically uint8.
- threshold1: Lower threshold for the hysteresis procedure.
- threshold2: Upper threshold for the hysteresis procedure. Strong edges above this are retained; weaker edges are considered based on connectivity.
Output:
- Returns a binary edge map of the same size as the input image, where:
- Edge pixels are 255 (white).
- Non-edge pixels are 0 (black).
- Data type: numpy.ndarray (uint8).

#### Thresholding and Binarization
`cv2.threshold`, `cv2.adaptiveThreshold` - Converts grayscale images to binary based on a threshold value. The former uses the same global threshold and the later calculates thresholds dynamically for smaller regions of the image.

- `cv2.threshold(img, thresh, maxval, type):` Applies binary or adaptive thresholding.
`

#### 2. **Morphological Transformations**
Helps in cleaning up binary images, filling gaps, or separating objects. Operates on binary images to remove noise or to emphasize specific shapes.

 `cv2.erode`, `cv2.dilate` erode basically thins the lines whereas dilate expands the lines. then you have open etc that do chains of these.
 `cv2.morphologyEx` : Applies advanced morphological transformations like opening, closing, gradient, etc., to process and clean up binary images.
 * Opening: Erosion followed by dilation (removes small noise)
 * Closing: Dilation followed by erosion (fills small gaps in objects)
 * Gradient: Highlights the edges of objects


#### 3. **Contour Detection an(d Analysis** 
Used for extracting and analyzing object boundaries. It requires a binary image, so thresholding or canny must have been applied before.

`cv2.findContours(image, mode, method)`: Contours are curves joining continuous points with the same intensity (e.g., boundaries of objects).\
- image: Source, an 8-bit single channel image
- mode: Contour retrieval mode. Options:
  - cv.RETR_EXTERNAL: only outer contours
  - cv.RETR_LIST: retrieve all countours with no regard of hierarchy
  - cv.RETR_CCOMP: Retrieves all contours and organized them into a two-level hierarchy
  - cv.RETR_TREE: Retrieves all the contours and constructs a full hierarchy
- method: Contour approximation method. NONE and SIMPLE. The later simply compresses lines into its two endpoints.


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

#### Others

filters: convolve arbitrary filters to blur ; sharpen etc.