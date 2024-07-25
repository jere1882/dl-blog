---
tag: Deep Learning Basics
aliases: 
publish: true
slug: deep-learning-specialization
title: Review and Summary of Deep Learning Specialization from DeepLearning
description: Notes, assignments and review of the course
date: 2024-07-09
image: /thumbnails/coursera_dl_specialization.png
---
## Introduction

My current employee gave me the chance to take the Deep Learning Specialization from DeepLearn.ai via Coursera. Even though this is quite a basic level course, I decided to give it a try and take the chance to revise my foundations.

There is quite a lot of undeserved praise online for this course. It lacked depth, there were no structured course notes (just videos), and the above all author kept embarking on lengthy "intuitive" explanations aimed at people who did not take a proper calculus course.

If you graduated from university, or if you already have some knowledge on neural networks, this course is probably not suitable for you. Nevertheless, even though I did not learn almost anything new, it was a great chance to revisit basic concepts and get a certificate that can be showcased in LinkedIn.

Below is a summary of the contents developed in the specialisation, based on my notes and takeaways.

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

This is mostly exemplified by the assignment 1 tasks, which you can check out in my GitHub.

#### Week 3: Shallow Neural Networks

##### Neural networks

There's a very verbose intro here just to explain the concept of a fully connected neural network and notation.

Basically it is a sequence of layers, each all consecutive layers are fully connected by parameters `w`.

![[Pasted image 20240717163408.png]]
*A neuron computes a weighted sum using a parameter `w` of all its parameters, adds a bias parameter `b` and then applies a non-linear activation function such as sigmoid*

We are introduced to different choices of activation functions. The rule of thumb is: use ReLu, unless you're on the last output layer of a binary classification task, there you can use Sigmoid.

Non-linearities are crucial to represent complex functions, otherwise the entire network collapses into a single linear function!

![[Pasted image 20240717164234.png]]

Here's a FCN with two layers (the input layer doesn't count). The notation is defined as follows:
* `**z^\[i\]\_j**`: The output of the jth neuron at the ith layer.
* `**a^\[i\]\_j**`:  Activation function applied to the previous value
* `**w^\[i\]\_j**`:  weight to connect neurons in layer `{i-1}` to neuron `j` in layer `i`
* `**b^\[i\]\_j**`:  bias used by neuron j at layer i
These values are stacked together to form matrixes that allow for efficient vectorization:
 * `**a^\[i\]**` , `**z^\[i\]**` all activations from layer i
 * `**W^\[i\]**`: Weights that connect layer `{i-1}` to layer `{i}`
 * `**b^\[i\]**`: all biases of neurons at layer i
 
Thus we can rewrite the equations in vectorized form, where the (i) superscript means sample number `i`. We can stack all samples together to vectorize this as well.

![[Pasted image 20240717165639.png]]
##### Gradient Descent for Neural Netoworks

The author presents pseudocode for a single-hidden-layer neural network, and we can see it's quite similar to logistic regression:

![[Pasted image 20240717221009.png]]
The derivatives of the activation functions are:

* Sigmoid:
```
	g(z)  = 1 / (1 + np.exp(-z))
	g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
	g'(z) = g(z) * (1 - g(z))
```
* RELU
```
    g(z)  = np.maximum(0,z)
    g'(z) = { 0  if z < 0
              1  if z >= 0  }
```

The forward propagation goes as follows:

```
Z^1 = W1A0 + b1    # A^0 is X
A^1 = g1(Z1)
Z^2 = W2A1 + b2
A^2 = Sigmoid(Z2)   # Sigmoid because the output is between 0 and 1
```

Finally, let's do backpropagation:

```
dA^2 = -Y/A^2 + (1-Y)/(1-A^2)   # Calculus
dZ^2 = DL/DZ^2 = DL/DA^2 * DA^2/DZ^2 = dA^2 * (A^2 * (1-A^2)) = A^2 - Y^2
dW^2 =  dZ^2 * DZ^2/DW^2 = (dZ^2 * A^1.T)/m
db2 = Sum(dZ2) / m
dZ1 = (W2.T * dZ2) * g'1(Z1)  # element wise product (*)
dW1 = (dZ1 * A0.T) / m   # A0 = X
db1 = Sum(dZ1) / m
```

![[Pasted image 20240717223618.png]]

Mira el video de la intuition por que:
* no entiendo por que involucra ms si la loss no es la J
* no entiendo por que transpuestas, asumo que algo de calculo de matrices
* no entiendo por que Sum en las derivadas de las bs

![[Pasted image 20240717223809.png]]
*Left: single sample - Right: All training set at once*

So basically you are calculating the direction you have to go to reduce the error for each training sample, and the you average out that direction to update the weights.
##### Random Initialization of Parameters
What happens if we initialize our NN to have all zero weights? 

It's possible to construct a proof by induction that if you initialize all the values of w to 0, each neuron in a given layer will compute the same output because they all start with the same weights and biases. Consequently, the gradients for these neurons will also be identical during backpropagation. This symmetry means that each neuron in a layer will continue to update in the same way, preventing the network from breaking symmetry and learning diverse features. Essentially, the neurons in each layer will remain identical throughout the training process, significantly limiting the model's capacity to learn.

The solution to this is to initialize the parameters randomly to small numbers, e.g.:

`W^1 = np.random.randn((2,2)) * 0.01`

Large numbers would cause activations such as sigmoid or tanh to saturate, making learning slow.

link to the assignment 3.

#### Week 4: Deep Neural Networks

A deep NN is a NN with three or more layers:
* `L`: Number of layers
* `n^[l]` : number of neurons at layer `l`  (`n[0]` number of inputs, `n[L]` number of outputs)
* `g^[l]` : activation function at layer `l`
* `z^[l] = W^[l] * a^[l-1] + b^[l]`   has shape `(n[l],1)`
* `a^[l] = g^[l](z^[l])`  : activation at the layer `l`  has shape `(n[l],1)
* `W^[l]`: weights that connect layer `l-1` to layer `l` has shape `(n[l],n[l-1])`
* `b^[l]`: biases of layer `l`, shape is `(n[l],1)`
* `x = a[0]`, `a[l] = y'`

Vectorized for the entire training set:
* `m`: number of training samples
* `Z^[i] = W^[i] X + b^[i]` has shape `(n[i],m)`
* `X` has shape `(n[0],m)` (i.e. each sample is a column, contrary to the conventions that the rest of the world uses)
* `A[0]=X`
* `A[l]` = `g^[l](Z^[l])` has shape  `(n[i],m)`
* `dZ^[l]` and `dA^[l]` will have shape `(n[i],m)`

Why use deeper networks? earlier layers of the neural network can detecting simple functions, like edges and then
composing them together in the later layers of a neural network so that it can learn more and more complex functions.

This is the typical intuition for hierarchical feature extraction in CNNs (I don't actually think it applies to dense layers though)

![[Pasted image 20240718162825.png]]

Another example with speech recognition. The first level of a NN may detect low lavel audio wave forms and basic phonemes. Later layers may compose this together.

This in some way emulates what our brain does.
##### Generic forward and backward functions (vectorized to all samples)

- Pseudo code for forward propagation for layer l:

    ```
    Input  A[l-1]
    Z[l] = W[l]A[l-1] + b[l]
    A[l] = g[l](Z[l])
    Output A[l], cache(Z[l])
    ```

- Pseudo code for back propagation for layer l:
    ```
    Input da[l], Caches
    dZ[l] = dA[l] * g'[l](Z[l])
    dW[l] = (dZ[l]A[l-1].T) / m
    db[l] = sum(dZ[l])/m                # Dont forget axis=1, keepdims=True
    dA[l-1] = w[l].T * dZ[l]            # The multiplication here are a dot product.
    Output dA[l-1], dW[l], db[l]
    ```

- Cross entropy loss function is:
    ```
    dA[L] = (-(y/a) + ((1-y)/(1-a)))
    ```

##### Parameters and Hyperparameters
* The parameters of the network are `w` and `b` and they are optimized via backpropagation
* Hyperparameters are higher level parameters of the algorithm such as:
	* The learning rate (alpha)
	* The number of iterations until convergence
	* The number of hidden layers `L` and the number of hidden units per hidden layer
	* The choice of activation functions

Hyperparameters are often optimized by trying a few combinations of them and checking which one works better (this is actually developed in depth in the next course)

I cannot recommend this video and its sequel more:  https://www.youtube.com/watch?v=Ilg3gGewQ5U.

## Course 2 - Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

### Week 1: Practical Aspects of Deep Learning

#### Train / Validation / Test set

In machine learning, we usually iterate through the loop:

`Idea -> Code -> Experiment -> Repeat`

During the process of defining the hyperparameters of a model, it is recommended that the available data is split into three sets:

| Data partition             | Purpose                                       | Size       |
| -------------------------- | --------------------------------------------- | ---------- |
| Training set               | Optimizing model parameters                   | 60 to 98 % |
| Validation set AKA dev set | Optimizing model hyperparameters              | 20 to 1%   |
| Test set                   | Provide an unbiased estimation of performance | 20 to 1%   |
If size of the dataset is greater than 1M we may use 98/1/1 or even 99.5/0.25/0.25

* Train and validation set must come from the same distribution
* The dev set is used to try out trained models with different hyperparameters, in order to define what is the best hyperparameter configuration
* It is ok to only have validation set without testing set if you don't need an unbiased estimation of performance. People call the validation set "test set" in this setup (although it is misleading).
#### Bias and Variance

![[Pasted image 20240725143318.png]]
* A model that underfits has high bias
	* e.g. training error doesn't go down near the optimal classifier error
* A model that overfits has high variance
	* big gap between training and validation error.
* A good model strikes a good balance between bias an variance
	* training and validation error closer to the optimal classifier

A basic recipe for Machine Learning is:

1. If your algorithm has high bias
	1. Make your NN larger (more hidden units or layers) - This never hurts. Still I advise starting smaller.
	2. Try out different models
	3. Train for longer
	4. Use a different optimization algorithm
2. If your algorithm has high variance
	1. Add more data
	2. Regularization
	3. A different model

Historical note: In traditional ML we usually talk about the **bias/variance tradeoff**, but in the deep learning world we can usually try to improve either without making the other worse.

#### Regularization

Regularization techniques reduce overfitting and therefore reduce variance. 

We can apply regularization to the **parameters** of the model by penalising large weights in the cost function. In order words, we try to keep the model from becoming "too complex".

The cost function looks like this when we add regularization, where lambda is the regularization hyperparameter:

`J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||)`

We can use either L1 or L2 norm:
* L1 makes the weights sparse, potentially making for a smaller and more efficient model.
* L2 is much more popular.













