---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: performance-metrics
title: Performance metrics
description: Assorted notes and takeaways
date: 2024-11-26
image: /thumbnails/backbones.png
---
# Binary classification

The following metrics are commonly used to assess the performance of classification models:

| **Metric**         | **Formula**                                   | **Interpretation**                          |
|---------------------|-----------------------------------------------|---------------------------------------------|
| **Accuracy**        | \(\frac{TP + TN}{TP + TN + FP + FN}\)        | Overall performance of the model            |
| **Precision**       | \(\frac{TP}{TP + FP}\)                       | How accurate the positive predictions are   |
| **Recall (Sensitivity)** | \(\frac{TP}{TP + FN}\)                   | Coverage of actual positive samples         |
| **Specificity**     | \(\frac{TN}{TN + FP}\)                       | Coverage of actual negative samples         |
| **F1 Score**        | \(\frac{2TP}{2TP + FP + FN}\)                | Hybrid metric useful for unbalanced classes |

The receiver operating characteristic (ROC) curve plots TPR versus FPR by varying the threshold. These metrics are summarized below:

![[Pasted image 20241220174613.png]]

| **Metric**                    | **Formula**            | **Equivalent**           |     |
| ----------------------------- | ---------------------- | ------------------------ | --- |
| **True Positive Rate (TPR)**  | \$\frac{TP}{TP + FN}\$ | Recall, Sensitivity      |     |
| **False Positive Rate (FPR)** | $\frac{FP}{TN + FP}$   | $1 - \text{Specificity}$ |     |

| **Metric**                     | **Formula/Definition**                                                    | **Interpretation**                                                                 |
|---------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Area Under the ROC Curve (AUC-ROC)** | Integral of the ROC curve                                              | Overall ability to rank positive samples higher than negatives across thresholds |
| **Area Under the Precision-Recall Curve (AUC-PR)** | Integral of the Precision-Recall curve                                | Focuses on the performance of the positive class                                 |

- **AUC-ROC** scores range from 0.5 (random guessing) to 1 (perfect discrimination).
- **AUC-PR** is particularly valuable when there’s a significant imbalance between classes, as it emphasizes the minority class.

Both metrics are widely used in fields like fraud detection, medical diagnosis, and any domain with highly skewed class distributions.

# Multi-class classification
metrics are extended to handle multiple classes

| **Metric**                       | **Description**                                                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Accuracy**                     | Fraction of correctly classified samples over the total.                                            |
| **Precision (Macro)**            | Average precision across all classes, treating each class equally.                                  |
| **Precision (Weighted)**         | Precision weighted by the number of samples in each class.                                          |
| **Recall (Macro)**               | Average recall across all classes, treating each class equally.                                     |
| **Recall (Weighted)**            | Recall weighted by the number of samples in each class.                                             |
| **F1-Score (Macro)**             | Average F1-score across all classes, treating each class equally.                                   |
| **F1-Score (Weighted)**          | F1-score weighted by the number of samples in each class.                                           |
| **AUC-ROC (Macro)**              | Average AUC-ROC over all classes using one-vs-rest (OvR).                                           |
| **AUC-ROC (Weighted)**           | AUC-ROC weighted by the number of samples in each class.                                            |
| **Mean Average Precision (mAP)** | Average of the areas under the precision-recall curve for each class (macro or weighted averaging). |
| **Top-K Accuracy**               | Fraction of samples where the true class is among the top-K predicted classes.                      |
|                                  |                                                                                                     |
# Regression

# Metrics for Regression

| **Metric**                                | **Formula**                                                                                       | **Interpretation**                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Mean Absolute Error (MAE)**             | $ \frac{1}{n} \sum_{i=1}^n \text{abs}(y_i - \hat{y}_i) $                                          | Average magnitude of errors, providing a linear measure of error without considering direction.        |
| **Mean Squared Error (MSE)**              | $ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $                                                  | Penalizes larger errors more heavily, focusing on large deviations from the actual values.             |
| **Root Mean Squared Error (RMSE)**        | $ \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} $                                           | Square root of MSE, interpreted in the same units as the target variable.                              |
| **R-squared (R²)**                        | $ 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $                   | Proportion of variance in the dependent variable explained by the model.                               |
| **Adjusted R²**                           | $ 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1} $                                                        | R² adjusted for the number of predictors in the model; avoids overestimating with multiple predictors. |
| **Mean Absolute Percentage Error (MAPE)** | $ \frac{1}{n} \sum_{i=1}^n \text{abs}\left(\frac{y_i - \hat{y}_i}{y_i}\right) \times 100 $        | Percentage-based metric showing the average error as a percentage of the actual values.                |
| **Median Absolute Error**                 | Median of $ \text{abs}(y_i - \hat{y}_i) $                                                         | Robust to outliers; represents the middle absolute error value.                                        |
| **Explained Variance**                    | $ 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)} $                                             | Measure of the discrepancy between predicted and actual variances.                                     |
| **Huber Loss**                            | Combines MSE for small errors and MAE for large errors, controlled by a hyperparameter $ \delta $ | Useful for datasets with outliers; balances the sensitivity between MAE and MSE.                       |


# Metrics for Object Detection

| **Metric**                    | **Formula**                                                                                                 | **Interpretation**                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Intersection over Union (IoU)** | $ \frac{\text{Area of overlap between predicted and ground truth boxes}}{\text{Area of union}} $           | Measures the overlap between predicted bounding box and ground truth.                                       |
| **Precision**                 | $ \frac{\text{True Positives}}{\text{True Positives + False Positives}} $                                    | Measures how accurate the detected objects are.                                                             |
| **Recall**                    | $ \frac{\text{True Positives}}{\text{True Positives + False Negatives}} $                                    | Measures how many ground truth objects are detected.                                                        |
| **Mean Average Precision (mAP)** | Average precision over all classes and IoU thresholds.                                                    | Evaluates overall object detection performance across all classes.                                           |
| **AP@[IoU=0.5]**              | $ \text{AP calculated at IoU threshold of 0.5} $                                                           | Evaluates precision and recall at a 0.5 IoU threshold.                                                      |
| **AP@[IoU=0.5:0.95]**         | $ \text{Average precision across IoU thresholds from 0.5 to 0.95 (step=0.05)} $                            | Evaluates robustness across multiple IoU thresholds.                                                        |
| **Per-Class mAP**             | $ \frac{1}{N} \sum_{c=1}^N \text{AP for class } c $                                                        | Average precision for each class, considering imbalance between classes.                                     |
| **Detection Time**            | Time taken to detect objects in an image.                                                                   | Measures inference speed, useful for real-time applications.                                                |
| **Number of False Positives** | Count of detected boxes that do not match any ground truth.                                                 | Indicates over-detection errors.                                                                             |
| **Number of False Negatives** | Count of ground truth objects missed by the detector.                                                       | Indicates under-detection errors.                                                                            |
| **AR (Average Recall)**       | Average recall calculated at various IoU thresholds.                                                        | Summarizes how well the model retrieves ground truth objects across IoU thresholds.                         |



# Metrics for Semantic Segmentation

| **Metric**                    | **Formula**                                                                                                 | **Interpretation**                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Pixel Accuracy**            | $ \frac{\text{Number of correctly predicted pixels}}{\text{Total number of pixels}} $                      | Fraction of correctly classified pixels.                                                                    |
| **Mean Pixel Accuracy**       | $ \frac{1}{N} \sum_{c=1}^N \frac{\text{Correct predictions for class } c}{\text{Total pixels in class } c} $| Average accuracy for each class, accounting for class imbalance.                                            |
| **Intersection over Union (IoU)** | $ \frac{\text{True Positive}}{\text{True Positive + False Positive + False Negative}} $                   | Overlap between predicted and ground truth areas divided by their union.                                    |
| **Mean IoU (mIoU)**           | $ \frac{1}{N} \sum_{c=1}^N \text{IoU for class } c $                                                       | Average IoU across all classes.                                                                             |
| **Dice Coefficient (F1 Score)** | $ \frac{2 \cdot \text{True Positive}}{2 \cdot \text{True Positive} + \text{False Positive} + \text{False Negative}} $ | Measures overlap between prediction and ground truth, emphasizing smaller regions.                          |
| **Boundary IoU**              | Computes IoU specifically for boundaries of segmented regions.                                              | Focuses on the accuracy of object boundaries.                                                               |
| **Per-Class Precision**       | $ \frac{\text{True Positive for class}}{\text{True Positive + False Positive for class}} $                  | Measures how well each class is correctly predicted relative to predictions.                                 |
| **Per-Class Recall**          | $ \frac{\text{True Positive for class}}{\text{True Positive + False Negative for class}} $                  | Measures how well each class is detected.                                                                   |

# Loss functions
| Loss Name              | Formula                                                                                 | Use Case                                              |
|-------------------------|-----------------------------------------------------------------------------------------|-------------------------------------------------------|
| Mean Squared Error (MSE) | $MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$                                   | Regression tasks. Penalizes large errors heavily.     |
| Mean Absolute Error (MAE) | $MAE = \frac{1}{N} \sum_{i=1}^N \text{abs}(y_i - \hat{y}_i)$                          | Regression tasks. Robust to outliers.                |
| Huber Loss             | $\text{Huber}(a) = \begin{cases} \frac{1}{2} a^2 & \text{if } \text{abs}(a) \leq \delta \\ \delta (\text{abs}(a) - \frac{\delta}{2}) & \text{if } \text{abs}(a) > \delta \end{cases}$ | Regression with outlier tolerance and smooth gradients. |
| Cross-Entropy Loss     | $-\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C y_{i,k} \log(\hat{y}_{i,k})$                    | Multi-class classification.                          |
| Binary Cross-Entropy (BCE) | $-\frac{1}{N} \sum_{i=1}^N \big(y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\big)$ | Binary classification.                               |
| Kullback-Leibler Divergence | $\text{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$                          | Comparing probability distributions.                 |
| Hinge Loss             | $\text{Hinge} = \frac{1}{N} \sum_{i=1}^N \text{max}(0, 1 - y_i \hat{y}_i)$              | Binary classification with SVMs.                     |
| Focal Loss             | $-\frac{1}{N} \sum_{i=1}^N \alpha (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i)$           | Class-imbalanced classification tasks.               |
| IoU Loss               | $1 - \frac{\text{Intersection}}{\text{Union}}$                                          | Object detection or segmentation tasks.              |
| Dice Loss              | $1 - \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$             | Semantic segmentation tasks.                         |
| Smooth L1 Loss         | $\text{Smooth L1}(a) = \begin{cases} \frac{1}{2} a^2 & \text{if } \text{abs}(a) \leq 1 \\ \text{abs}(a) - \frac{1}{2} & \text{if } \text{abs}(a) > 1 \end{cases}$ | Object detection (e.g., bounding box regression).    |
