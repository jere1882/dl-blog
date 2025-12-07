---
has_been_reviewed: false
tag: Deep Learning Basics
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: metrics-and-losses
title: Performance metrics
description: Assorted notes and takeaways
date: 2024-11-26
image: /thumbnails/backbones.png
---
# Binary classification

The following metrics are commonly used to assess the performance of classification models:

| **Metric**         | **Formula**                                   | **Interpretation**                          |
|---------------------|-----------------------------------------------|---------------------------------------------|
| **Accuracy**        | $\frac{TP + TN}{TP + TN + FP + FN}$        | Overall performance of the model            |
| **Precision**       | $\frac{TP}{TP + FP}$                      | How accurate the positive predictions are   |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$                   | Coverage of actual positive samples         |
| **Specificity**     | $\frac{TN}{TN + FP}$                       | Coverage of actual negative samples         |
| **F1 Score**        | $\frac{2 * TP}{2 * TP + FP + FN}$                | Hybrid metric useful for unbalanced classes |

The receiver operating characteristic (ROC) curve plots TPR versus FPR by varying the threshold. These metrics are summarized below:

![[Pasted image 20241220174613.png]]

| **Metric**                    | **Formula**            | **Equivalent**           |
| ----------------------------- | ---------------------- | ------------------------ |
| **True Positive Rate (TPR)**  | $\frac{TP}{TP + FN}$ | Recall, Sensitivity      |
| **False Positive Rate (FPR)** | $\frac{FP}{TN + FP}$   | $1 - Specificity$ |

| **Metric**                     | **Formula/Definition**                                                    | **Interpretation**                                                                 |
|---------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Area Under the ROC Curve (AUC-ROC)** | Integral of the ROC curve                                              | Overall ability to rank positive samples higher than negatives across thresholds |
| **Area Under the Precision-Recall Curve (AUC-PR)** | Integral of the Precision-Recall curve                                | Focuses on the performance of the positive class                                 |

- **AUC-ROC** scores range from 0.5 (random guessing) to 1 (perfect discrimination).
- **AUC-PR** is particularly valuable when there's a significant imbalance between classes, as it emphasizes the minority class.

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

| **Metric**                                | **Interpretation**                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Mean Absolute Error (MAE)**             | Average magnitude of errors, providing a linear measure of error without considering direction.        |
| **Mean Squared Error (MSE)**              | Penalizes larger errors more heavily, focusing on large deviations from the actual values.             |
| **Root Mean Squared Error (RMSE)**        | Square root of MSE, interpreted in the same units as the target variable.                              |
| **R-squared (R²)**                        | Proportion of variance in the dependent variable explained by the model.                               |
| **Adjusted R²**                           | R² adjusted for the number of predictors in the model; avoids overestimating with multiple predictors. |
| **Mean Absolute Percentage Error (MAPE)** | Percentage-based metric showing the average error as a percentage of the actual values.                |
| **Median Absolute Error**                 | Robust to outliers; represents the middle absolute error value.                                        |
| **Explained Variance**                    | Measure of the discrepancy between predicted and actual variances.                                     |
| **Huber Loss**                            | Useful for datasets with outliers; balances the sensitivity between MAE and MSE.                       |

**Mean Absolute Error (MAE):**  
$ \frac{1}{n} \sum_{i=1}^n abs(y_i - \hat{y}_i) $

**Mean Squared Error (MSE):**  
$ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $

**Root Mean Squared Error (RMSE):**  
$ sqrt( \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 ) $
# Metrics for Object Detection

| **Metric**                        | **Formula**                                                            | **Interpretation**                                                                  |
| --------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Intersection over Union (IoU)** | Area of overlap between predicted and ground truth boxes/Area of union | Measures the overlap between predicted bounding box and ground truth.               |
| **Precision**                     | $ \frac{TP}{TP+FP} $                                                   | Measures how accurate the detected objects are.                                     |
| **Recall**                        | $ \frac{TP}{TP+FN} $                                                   | Measures how many ground truth objects are detected.                                |
| **Mean Average Precision (mAP)**  | Average precision over all classes and IoU thresholds.                 | Evaluates overall object detection performance across all classes.                  |
| **AP@[IoU=0.5]**                  | AP calculated at IoU threshold of 0.5                                  | Evaluates precision and recall at a 0.5 IoU threshold.                              |
| **AP@[IoU=0.5:0.95]**             | Average precision across IoU thresholds from 0.5 to 0.95 (step=0.05)   | Evaluates robustness across multiple IoU thresholds.                                |
| **Per-Class mAP**                 | $ \frac{1}{N} \sum_{c=1}^N (AP for class c) $                          | Average precision for each class, considering imbalance between classes.            |
| **Detection Time**                | Time taken to detect objects in an image.                              | Measures inference speed, useful for real-time applications.                        |
| **Number of False Positives**     | Count of detected boxes that do not match any ground truth.            | Indicates over-detection errors.                                                    |
| **Number of False Negatives**     | Count of ground truth objects missed by the detector.                  | Indicates under-detection errors.                                                   |
| **AR (Average Recall)**           | Average recall calculated at various IoU thresholds.                   | Summarizes how well the model retrieves ground truth objects across IoU thresholds. |



# Metrics for Semantic Segmentation

| **Metric**                    | **Formula**                                                                                                 | **Interpretation**                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Pixel Accuracy**            | (Number of correctly predicted pixels)/(Total number of pixels)                      | Fraction of correctly classified pixels.                                                                    |
| **Mean Pixel Accuracy**       | $ \frac{1}{N} \sum_{c=1}^N (Correct predictions for class c) / (Total pixels in class  c) $| Average accuracy for each class, accounting for class imbalance.                                            |
| **Intersection over Union (IoU)** | $ \frac{TP}{TP+FP+FN} $                   | Overlap between predicted and ground truth areas divided by their union.                                    |
| **Mean IoU (mIoU)**           | $ \frac{1}{N} \sum_{c=1}^N IoU_c $                                                       | Average IoU across all classes.                                                                             |
| **Dice Coefficient (F1 Score)** | $ \frac{2 * TP}{2 * TP + FP + FN} $ | Measures overlap between prediction and ground truth, emphasizing smaller regions.                          |
| **Boundary IoU**              | Computes IoU specifically for boundaries of segmented regions.                                              | Focuses on the accuracy of object boundaries.                                                               |
| **Per-Class Precision**       | $ \frac{TP_c}{TP_c + FP_c} $                  | Measures how well each class is correctly predicted relative to predictions.                                 |
| **Per-Class Recall**          | $ \frac{TP_c}{TP_c + FN_c} $                  | Measures how well each class is detected.                                                                   |

# Loss functions
| Loss Name                                           | Formula                                                                                 | Use Case                                              |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Mean Squared Error (MSE)                            | $MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$                                   | Regression tasks. Penalizes large errors heavily.     |
| Mean Absolute Error (MAE)                           | $MAE = \frac{1}{N} \sum_{i=1}^N abs(y_i - \hat{y}_i)$                          | Regression tasks. Robust to outliers.                |
| Huber Loss                                          |  google it | Regression with outlier tolerance and smooth gradients. |
| Cross-Entropy Loss                                  | $-\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C y_{i,k} \log(\hat{y}_{i,k})$                    | Multi-class classification.                          |
| Binary Cross-Entropy (BCE)                          | $-\frac{1}{N} \sum_{i=1}^N \big(y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\big)$ | Binary classification.                               |
| Kullback-Leibler Divergence                         | $KL(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$                          | Comparing probability distributions.                 |
| Hinge Loss                                          | $Hinge = \frac{1}{N} \sum_{i=1}^N max(0, 1 - y_i \hat{y}_i)$              | Binary classification with SVMs.                     |
| Focal Loss                                          | $-\frac{1}{N} \sum_{i=1}^N \alpha (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i)$           | Class-imbalanced classification tasks.               |
| IoU Loss                                            | $1 - \frac{Intersection}{Union}$                                          | Object detection or segmentation tasks.              |
| Dice Loss                                           | $1 - \frac{2 * TP}{2 * TP + FP + FN} $             | Semantic segmentation tasks.                         |
| Smooth L1 Loss                                      | next | Object detection (e.g., bounding box regression).    |
# A deep dive into Cross Entropy

## Cross entropy for model evaluation

Cross-entropy measures the difference between two probability distributions, usually:
* the true distribution: what should happen
* the model's predicted distribution: what the model thinks will happen
Cross-entropy answers:
*How surprised is the model by the actual outcome?*

### Cross entropy for a single sample:

For classification, let's say there are K possible classes. Then the cross entropy is:

![[Pasted image 20250616215344.png]]
For a single model prediction q and ground truth probability p. 

E.g. 
task: predicting the class of an input(cat - dog - fish) 
for a given input, `p=[1,0,0]` (true class is cat) and `q=[0.8,0.1,0.1]` (model prediction)

Then 

H(p,q) = - log 0.8

Observations:
* The log in machine learning is always `ln` base e because the gradient is simple: `1/x`
![[Pasted image 20250616220050.png]]
* the log is applied to the model prediction:
	* p * q ->linear, the penalty is directly proportional to how good the prediction is
	* -p * log(q) -> non linear. it explodes when the model is wrong! applies thus a stronger penalty - this is because   `x << -log(x)` when x is small
	* Example: 
		* K=3 ; `p=[0,1,0]` ; `q1=[0.9,0.05,0.05]`; `q2=[0.4,0.3,0.3]`

| **Case**               | **Prediction qq**    | **Dot Product (p · q)** | **Cross-Entropy Loss** −log⁡q(c)-\log q(c) | **Better?**                                        |
| ---------------------- | -------------------- | ----------------------- | ------------------------------------------ | -------------------------------------------------- |
| **Confidently Wrong**  | `[0.90, 0.05, 0.05]` | **0.05**                | **2.9957**                                 | ❌ Bad — Confident but wrong → Strong penalty in CE |
| **Uncertain / Unsure** | `[0.40, 0.30, 0.30]` | **0.30**                | **1.2040**                                 | ✅ Less bad — unsure, gentler penalty               |
		* Remember that the loss should be LARGE for incorrect predictions. The dot product is actually terrible, it penalizes less the confidently wrong one!
	* Cross entropy is 0 for a perfect prediction and it's always >= 0 
	* Simple MSE is not good for classes or probabilities because it punishes large numeric errors but not uncertainty. In the above example, MSE for the confidently wrong is 0.2 and MSE for the uncertian is much larger.
	* Cross entropy is used as a loss because it's differentiable, smooth and provides very useful gradient signals. Then at evaluation time, confusion-matrix derived metrics are discrete, interpretable performance metrics more understandable for humans.

### Cross entropy for an entire dataset
Is simply averaging the cross entropy for each sample in the dataset.
## Empirical cross-entropy (Data entropy)

Given a probability distribution, entropy is a measure of uncertainty or surprise in a probability distribution:
![[Pasted image 20250616230214.png]]
* if the distribution `p` is uniform (`p(x_i) = 1/N`) the entropy is maximized. Every outcome is equally likely, maximum uncertainty.
* if the distribution is peaked (one outcome has all probability), entropy is zero - There is no uncertainty.
* Entropy is a property of the distribution itself.

In the context of ML, we can talk of the entropy of a dataset if we define the distribution of it's labels, values or features.

E.g:  
* 50% samples of class A ; 50% samples class =B => H=1
* Entropy of the labels => how varied the class labels are
* Entropy of features => how predictable or random the feature values are
* Joint entropy => how unpredictable are the pairs feature/labels

In this context, cross entropy is **not "error" or "loss"** — it's a statistical property describing **uncertainty/diversity in the data itself**.

