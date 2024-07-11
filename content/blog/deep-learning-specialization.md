---
tag: Deep Learning Basics
aliases: 
publish: true
slug: deep-learning-specialization
title: Review and Summary of Deep Learning Specialization from DeepLearning.ai
description: Notes, assignments and review of the course
date: 2024-07-09
image: /thumbnails/pick_architecure.jpeg
---
## Introduction

My current employee gave me the chance to get the Deep Learning Specialization from DeepLearn.ai via Coursera. 

In this post I will write down my take-aways and notes from each section.
(make a valuation later on)

## Course 1: Neural Networks and Deep Learning

#### Week 1: Introduction to Deep Learning

This is a very relaxed welcome unit. Neural networks are loosely presented as systems that can learn to solve tasks involving many inputs and outputs, and excel at dealing with unstructured data (images, audio, text)

Structured data, on the other hand,  is highly organized and easily searchable. It is typically stored in relational databases or spreadsheets, where the data is arranged in rows and columns with specific data types.

The teacher emphasizes that neural networks performance scale with the amount of data, and in general are very data-hungry. They are able to surprass traditional ML methods (such as SVM), but they do require substantial quantity of data samples.

#### Week 2: Basics of Neural Network programming

##### Logistic Regression

Binary classification is exemplified with the task of predicting, giving an input image, whether it is a cat picture or a doc picture. 

In this example, RGB images of 64x64 pixels are rearranged into a flat vector of 12288 elements. This is the dimension of the inputs in this setup for this task.

Given a dataset of `m` cat-dog pictures and their labels, they are arranged into matrixes as follows:

X:  A `n` x `m` matrix where each column is one of the training inputs, and `n` is the leng th of the inputs (12288 for this example)
Y: A `1` x `m`  matrix where each column is the label of one of the training inputs.

The teaches introduced a well known, simple classification method: Linear Regression. 

Given an input `x` (e.g. a dog-cat image), it is meant to predict `\hat(y)`  where `0 <= \hat(y) <= 1` by applying a linear function followed by a sigmoid function to squeeze the output of the linear function to `[0,1]`

![[Pasted image 20240709193121.png]]
The parameters of the model are `w` and `b`. Recall the shape of the sigmoid function here:

![[Pasted image 20240709193224.png]]
In order to use the training set of `m` labeled samples to tune the parameters `w` and `b` , a few concepts are introduced:

* The **loss** (error) function **L**, which tells how different a model prediction is from its labelled ground truth. Since this is a binary classification problem, the **Binary Cross-Entropy Loss is introduced**

![[Pasted image 20240709193651.png]]
* The **cost** function **J** is the sum of the loss over the entire training set for any given choice of parameters
![[Pasted image 20240709194013.png]]
* **Gradient descent** is an iterative method used to find w and b that minimize J(w,b). It consists of updating w and b on each iteration by substracting the derivative of J with respect to the parameters. By moving the paremeters in the direction opposite to the gradient, the cost function value is driven down and the model should become better at predicting the training labels. A hyperparameter **alpha**, the learning rate, governs the strength of the parameter update on each iteration.

Here, the teacher goes into A LOT of extra-boring examples trying to impress the meaning of a derivative without actually diving into the math. Having taking in-depth courses of calculus at university, this was a stump.

It's actually quite simple. `f'(a)` gives an approximation of how `f` changes if you slightly modify `a`. Say `f'(a)=2`, then `f'(a+delta) ~= f(a) + 2*delta`
 
It becomes a little more interesting when we talk about derivatives in a computational graph. Here a system takes three inputs, `a`, `b` and `c` ; makes a few intermediate computations `u` and `v` and finally produces an output `J`

![[Pasted image 20240709195439.png]]

The key point here is that, if we want to know the derivative of `J` with respect to the inputs (e.g. `b` or `c`), we need to apply the **chain rule** and go through the intermediate nodes.

![[Pasted image 20240709195647.png]]

In the context of logistic regression, we can calculate the derivative of the loss L by writing it as a computational graph:

![[Pasted image 20240709195909.png]]

This example has inputs with size 2 (thus two weights). We need the derivative of `L` wrt to the parameters `w_1`, `w_2` and `b`.

In order to do so, we first need the intermediate derivatives:

`da = D L(a,y) / D a = -y/a + (1-y)/(1-a)`
`dz = DL/Dz = DL/Da * Da/Dz = da * a*(1-a) = a-y`
`dw1 = DL/dw1 = DL/z * Dz/Dw1 = dz * x1`

After calculating the final derivative, we can update
`w1 := w1 - alpha * dw1`

And do the same for the other parameters.

To scale this up to `m` examples, we need to calculate the derivatives of the cost function `J`. Since the derivative of a sum is the sum of the derivatives, this is quite straightforward.
![[Pasted image 20240709203724.png]]

This section concludes with a snippet of code that makes an iteration of gradient descent, calculating the cumulative gradient over the entire training set. Notice that this would have to be repeated iteratively until convergence.

Also notice that n=2 meaning that x is just x1 and x2. If n is large, this code is inefficient. Vectorization would have to be used to get rid of for loops.

![[Pasted image 20240709211734.png]]

##### Vectorization
Replacing explicit `for` loops by matrix/vector built in operations can considerable speed up your algorithms both in GPU and CPU.

Examples:

* In order to multiply a matrix by a vector, use `np.dot(A,v)`
* In order to apply the same function to each element in a vector `v`, use built-in operations that are very efficient `np.exp(v)`, `np.log(v)`, `np.abs(v)`, `np.maximum(v,0)`,  `v**2`, `1/v`
* use `np.sum` , 

I think it's just better to go directly to the assignment



**TODO AT THE END: Move all the assignments to github.**



